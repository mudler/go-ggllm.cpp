#include "binding.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "falcon_common.h"
#include "libfalcon.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#include <shellapi.h>
#endif

static console_state con_st;
static falcon_context ** g_ctx;

int falcon_predict(void* params_ptr, void* state_pr, char* result, bool debug) {
    gpt_params* params_p = (gpt_params*) params_ptr;
    falcon_context* ctx = (falcon_context*) state_pr;
  
    g_ctx = &ctx;
    gpt_params params = *params_p;

       if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    std::mt19937 rng(params.seed);

    std::vector<falcon_token> embd_inp; // tokenized prompt
    std::vector<falcon_token> inp_system = {}; // system prompt
    std::vector<falcon_token> inp_pfx = {}; // prefix to user prompt
    std::vector<falcon_token> inp_sfx = {}; // suffix to user prompt
    std::vector<std::vector<falcon_token>> stopwords = {};


    if (params.stopwords.size())
    {
        std::string sw_token_str;
        std::vector<std::string> inp_system;
        std::stringstream stopwordStream(params.stopwords);
        std::vector<std::string> sw_token_list;
        while(std::getline(stopwordStream, sw_token_str, ',')) {
            sw_token_list.push_back(sw_token_str);
        }

        for (auto& sw_token : sw_token_list) {
            auto stopword_seq = ::falcon_tokenize(ctx, sw_token, false);
            stopwords.push_back(stopword_seq);
        }
    }



    std::string path_session = params.path_prompt_cache;
    std::vector<falcon_token> session_tokens;

    if (!path_session.empty()) {
        fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(params.n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            llama_set_rng_seed(ctx, params.seed);

            fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
        }
    }

    // tokenize the prompt
    

    if (params.interactive_first || params.instruct || !params.prompt.empty() || session_tokens.empty()) 
    {
        // Falcon does not have a dedicated bos token (bos==eos), so don't inject it here
        // auto start = ggml_time_us();
        embd_inp = ::falcon_tokenize(ctx, params.prompt, false);
        // auto end = ggml_time_us();
        // fprintf(stderr, "%s: tokenization took %0.3f ms\n", __func__, (end - start) / 1000.0);
        // fprintf(stderr, "%s: tokens processed: %d\n", __func__, (int) embd_inp.size());
        // fprintf(stderr, "%s: tokens/second : %0.3f\n", __func__, (embd_inp.size() / ((end - start) / 1000000.0)));
    } else {
        embd_inp = session_tokens;
    }

    const int n_ctx = falcon_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }
    falcon_prepare_buffers(ctx, params.n_batch, (int)(embd_inp.size()+1));
    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size()) {
        for (falcon_token id : session_tokens) {
            if (params.verbose_prompt)
            {
                const char *c_tk = falcon_token_to_str(ctx, id);
                if (*c_tk == '\n') c_tk="\\n";
                if (*c_tk == '\r') c_tk="\\r";
                fprintf(stderr, "SESSION TOKEN MATCH: %6d -> '%s'\n", id, c_tk);
            }
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                if (params.verbose_prompt)
                {
                    const char *c_tk = falcon_token_to_str(ctx, id);
                    if (*c_tk == '\n') c_tk="\\n";
                    if (*c_tk == '\r') c_tk="\\r";
                    fprintf(stderr, "SESSION TOKEN MISMATCH: %6d -> '%s'\n", id, c_tk);
                }
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            fprintf(stderr, "%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
    }

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() &&
            session_tokens.size() > embd_inp.size()) {
        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // determine newline token
    //auto llama_token_newline = ::falcon_tokenize(ctx, "\n", false);
    auto falcon_token_newline = falcon_token_nl();

size_t prompt_size = embd_inp.size();

    // TODO: replace with ring-buffer
    std::vector<falcon_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<falcon_token> embd;

        std::string res = "";


    // do one empty run to warm up the model (doing this with a session would destroy first KV pair)
    if(n_matching_session_tokens <= 0)
    {
        const std::vector<falcon_token> tmp = { falcon_token_bos(), };
        falcon_eval(ctx, tmp.data(), (int)tmp.size(), 0, params.n_threads,0);
        llama_reset_timings(ctx);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) 
    {
        // predict
        if (embd.size() > 0) 
        {
            // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            auto max_embd_size = n_ctx - 4;
            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int)embd.size() > max_embd_size) {
                auto skipped_tokens = embd.size() - max_embd_size;
                console_set_color(con_st, CONSOLE_COLOR_ERROR);
                printf("<<input too long: skipped %zu token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
                fflush(stdout);
                embd.resize(max_embd_size);
            }

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            #if 1
            if (n_past + (int) embd.size() > n_ctx) 
            {
                const int n_left = n_past - params.n_keep;

                n_past = params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
                // stop saving session if we run out of context
                path_session.clear();

                //printf("\n---\n");
                //printf("resetting: '");
                //for (int i = 0; i < (int) embd.size(); i++) {
                //    printf("%s", falcon_token_to_str(ctx, embd[i]));
                //}
                //printf("'\n");
                //printf("\n---\n");
                if (params.verbose_prompt)
                {
                    fprintf(stderr, "\n#CONTEXT_RESET_START: ");
                    for (int i = 0; i < (int) embd.size(); i++) {
                        fprintf(stderr, "%d => %s, ", embd[i], falcon_token_to_str(ctx, embd[i]));
                    }
                    fprintf(stderr, " #RESET_END\n");
                }
            }
            #endif
            // New mode:
            /**
             * 1. n_keep needs to be set to the system prompt if one is used
             * 2. instead of reprocessing half of the context, we just cut the top parts of kv_cache without reprocessing
            */
            if (n_past + (int) embd.size() > n_ctx) 
            {
                
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }
            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) 
            {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }
                int debug_timings = params.debug_timings;
                if (n_remain == 1 && debug_timings == 2) debug_timings = 3; // we have no access to the last information in eval()
                if (falcon_eval(ctx, &embd[i], n_eval, n_past, params.n_threads,debug_timings)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return 1;
                }
                n_past += n_eval;
            }
            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = (int)session_tokens.size();
            }
        } // if (embd.size() > 0)

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed)  // sample for next generation
        {
            // out of user input, sample next token
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? falcon_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
            }

            falcon_token id = 0;

            {
                auto logits  = falcon_get_logits(ctx);
                auto n_vocab = falcon_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<falcon_token_data> candidates;
                candidates.reserve(n_vocab);
                for (falcon_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(falcon_token_data{token_id, logits[token_id], 0.0f});
                }

                falcon_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                // Apply penalties
                float nl_logit = logits[falcon_token_nl()];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    logits[falcon_token_nl()] = nl_logit;
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &candidates_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                        llama_sample_temperature(ctx, &candidates_p, temp);
                        id = llama_sample_token(ctx, &candidates_p);
                    }
                }
                // printf("`%d`", candidates_p.size);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode // todo: openassistant and wizard handling
            #if 0
            // disabled for now - some finetunes actually need that token - audit if that is needed by normal use
            if (id == falcon_token_eos() && params.interactive && !params.instruct) {
                id = falcon_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = ::falcon_tokenize(ctx, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }
            #endif

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            // call the token callback, no need to check if one is actually registered, that will
            // be handled on the Go side.
            auto token_str = falcon_token_to_str(ctx, id);
            
            if (!returntokenCallback(state_pr, (char*)token_str)) {
                break;
            }
        } else 
        {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) { // push input tokens into embd (n_batch)
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        for (auto id : embd) {
            res += falcon_token_to_str(ctx, id);
        }

        bool stopword_fulfilled = false;
        // stopwords
        if (!embd.empty()) 
        {
            
            for (const auto& stopword : stopwords) {
                if (embd.size() < stopword.size()) {
                    continue; // if embd is smaller than stopword, skip this iteration
                }
                stopword_fulfilled = true; // initially assume stopword is fulfilled
                for (size_t i = 0; i < stopword.size(); ++i) {
                    if (embd[embd.size() - i - 1] != stopword[stopword.size() - i - 1]) {
                        stopword_fulfilled = false;
                        break;
                    }
                }
                if (stopword_fulfilled) {
                    break;
                }
            }
            if (stopword_fulfilled) 
            {
                if (params.verbose_prompt) 
                    fprintf(stderr, " [stopword]\n");
                if (!params.interactive)
                    break;
            }
        }
      
        // end of text token
        if (!embd.empty() && embd.back() == falcon_token_eos() || stopword_fulfilled) 
        {
            
                if (params.verbose_prompt)
                    fprintf(stderr, " [end of text]\n");
                // if we are in the prompt ingestion we will not stop
                if (n_past > (int)embd_inp.size()) {
                    break;
                }
            
        }
        
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        fprintf(stderr, "\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    falcon_print_timings(ctx);
    strcpy(result, res.c_str()); 
    return 0;
}

void falcon_binding_free_model(void *state_ptr) {
    falcon_context* ctx = (falcon_context*) state_ptr;
    llama_free(ctx);
}

void falcon_free_params(void* params_ptr) {
    gpt_params* params = (gpt_params*) params_ptr;
    delete params;
}


std::vector<std::string> falcon_create_vector(const char** strings, int count) {
    std::vector<std::string>* vec = new std::vector<std::string>;
    for (int i = 0; i < count; i++) {
      vec->push_back(std::string(strings[i]));
    }
    return *vec;
}

void falcon_delete_vector(std::vector<std::string>* vec) {
    delete vec;
}

int falcon_load_state(void *ctx, char *statefile, char*modes) {
    falcon_context* state = (falcon_context*) ctx;
const falcon_context* constState = static_cast<const falcon_context*>(state);
    const size_t state_size = llama_get_state_size(state);
    uint8_t * state_mem = new uint8_t[state_size];

  {
        FILE *fp_read = fopen(statefile, modes);
        if (state_size != llama_get_state_size(constState)) {
            fprintf(stderr, "\n%s : failed to validate state size\n", __func__);
            return 1;
        }

        const size_t ret = fread(state_mem, 1, state_size, fp_read);
        if (ret != state_size) {
            fprintf(stderr, "\n%s : failed to read state\n", __func__);
            return 1;
        }

        falcon_set_state_data(state, state_mem);  // could also read directly from memory mapped file
        fclose(fp_read);
    }

    return 0;
}

void falcon_save_state(void *ctx, char *dst, char*modes) {
    falcon_context* state = (falcon_context*) ctx;

    const size_t state_size = llama_get_state_size(state);
    uint8_t * state_mem = new uint8_t[state_size];

    // Save state (rng, logits, embedding and kv_cache) to file
    {
        FILE *fp_write = fopen(dst, modes);
        falcon_copy_state_data(state, state_mem); // could also copy directly to memory mapped file
        fwrite(state_mem, 1, state_size, fp_write);
        fclose(fp_write);
    }
}

void* falcon_allocate_params(const char *prompt, int seed, int threads, int tokens, int top_k,
                            float top_p, float temp, float repeat_penalty, int repeat_last_n, bool ignore_eos, bool memory_f16, int n_batch, int n_keep, const char* stopwords,
                             float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap,
                             const char *maingpu,const char *tensorsplit , bool prompt_cache_ro) {
    gpt_params* params = new gpt_params;
    params->seed = seed;
    params->n_threads = threads;
    params->n_predict = tokens;
    params->repeat_last_n = repeat_last_n;
    params->prompt_cache_ro = prompt_cache_ro;
    params->top_k = top_k;
    params->top_p = top_p;
    params->memory_f16 = memory_f16;
    params->temp = temp;
    params->use_mmap = mmap;
    params->use_mlock = mlock;
    params->repeat_penalty = repeat_penalty;
    params->n_batch = n_batch;
    params->n_keep = n_keep;
    if (maingpu[0] != '\0') { 
        params->main_gpu = std::stoi(maingpu);
    }

    if (tensorsplit[0] != '\0') { 
        std::string arg_next = tensorsplit;
            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

            for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
                if (i < split_arg.size()) {
                    params->tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params->tensor_split[i] = 0.0f;
                }
            }  
    }

    params->prompt_cache_all = prompt_cache_all;
    params->path_prompt_cache = session_file;

    if (ignore_eos) {
        params->logit_bias[falcon_token_eos()] = -INFINITY;
    }
      params->stopwords = std::string(stopwords);
    
    params->tfs_z = tfs_z;
    params->typical_p = typical_p;
    params->presence_penalty = presence_penalty;
    params->mirostat = mirostat;
    params->mirostat_eta = mirostat_eta;
    params->mirostat_tau = mirostat_tau;
    params->penalize_nl = penalize_nl;
    std::stringstream ss(logit_bias);
    falcon_token key;
    char sign;
    std::string value_str;
    if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
        params->logit_bias[key] = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
    } 
    params->frequency_penalty = frequency_penalty;
    params->prompt = prompt;
    
    return params;
}


void* falcon_load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, bool mmap, bool vocab_only, int n_gpu_layers, int n_batch, const char *maingpu, const char *tensorsplit) {
    // load the model
    auto lparams = falcon_context_default_params();

    lparams.n_ctx      = n_ctx;
    lparams.seed       = n_seed;
    lparams.f16_kv     = memory_f16;
    lparams.embedding  = embeddings;
    lparams.use_mlock  = mlock;
    lparams.n_gpu_layers = n_gpu_layers;
    lparams.use_mmap = mmap;
    lparams.vocab_only = vocab_only;

    if (maingpu[0] != '\0') { 
        lparams.main_gpu = std::stoi(maingpu);
    }

    if (tensorsplit[0] != '\0') { 
        std::string arg_next = tensorsplit;
            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

            for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
                if (i < split_arg.size()) {
                    lparams.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    lparams.tensor_split[i] = 0.0f;
                }
            }  
    }

    lparams.n_batch      = n_batch;

    falcon_init_backend();
    void* res = nullptr;
    try {
        res = falcon_init_from_file(fname, &lparams);
    } catch(std::runtime_error& e) {   
        fprintf(stderr, "failed %s",e.what());
        return res;
    }

    return res;
}
