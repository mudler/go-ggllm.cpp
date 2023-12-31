package ggllm

// #cgo CXXFLAGS: -I${SRCDIR}/ggllm.cpp/examples -I${SRCDIR}/ggllm.cpp
// #cgo LDFLAGS: -L${SRCDIR} -lggllm -lm -lstdc++
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo darwin CXXFLAGS: -std=c++11
// #include "falcon_binding.h"
import "C"
import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"
)

type Falcon struct {
	state       unsafe.Pointer
	embeddings  bool
	contextSize int
}

func New(model string, opts ...ModelOption) (*Falcon, error) {
	mo := NewModelOptions(opts...)
	modelPath := C.CString(model)
	result := C.falcon_load_model(modelPath, C.int(mo.ContextSize), C.int(mo.Seed), C.bool(mo.F16Memory), C.bool(mo.MLock), C.bool(mo.Embeddings), C.bool(mo.MMap), C.bool(mo.VocabOnly), C.int(mo.NGPULayers), C.int(mo.NBatch), C.CString(mo.MainGPU), C.CString(mo.TensorSplit))
	if result == nil {
		return nil, fmt.Errorf("failed loading model")
	}

	ll := &Falcon{state: result, contextSize: mo.ContextSize, embeddings: mo.Embeddings}

	return ll, nil
}

func (l *Falcon) Free() {
	C.falcon_binding_free_model(l.state)
}

func (l *Falcon) LoadState(state string) error {
	d := C.CString(state)
	w := C.CString("rb")

	result := C.falcon_load_state(l.state, d, w)
	if result != 0 {
		return fmt.Errorf("error while loading state")
	}

	return nil
}

func (l *Falcon) SaveState(dst string) error {
	d := C.CString(dst)
	w := C.CString("wb")

	C.falcon_save_state(l.state, d, w)

	_, err := os.Stat(dst)
	return err
}

func (l *Falcon) Predict(text string, opts ...PredictOption) (string, error) {
	po := NewPredictOptions(opts...)

	if po.TokenCallback != nil {
		setReturnCallback(l.state, po.TokenCallback)
	}

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	params := C.falcon_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), C.CString(po.StopPrompts),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
	)
	ret := C.falcon_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])), C.bool(po.DebugMode))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")
	res = strings.TrimSuffix(res, "<|endoftext|>")

	C.falcon_free_params(params)

	if po.TokenCallback != nil {
		setReturnCallback(l.state, nil)
	}

	return res, nil
}

// CGo only allows us to use static calls from C to Go, we can't just dynamically pass in func's.
// This is the next best thing, we register the callbacks in this map and call tokenCallback from
// the C code. We also attach a finalizer to LLama, so it will unregister the callback when the
// garbage collection frees it.

// SetTokenCallback registers a callback for the individual tokens created when running Predict. It
// will be called once for each token. The callback shall return true as long as the model should
// continue predicting the next token. When the callback returns false the predictor will return.
// The tokens are just converted into Go strings, they are not trimmed or otherwise changed. Also
// the tokens may not be valid UTF-8.
// Pass in nil to remove a callback.
//
// It is save to call this method while a prediction is running.
func (l *Falcon) SetTokenCallback(callback func(token string) bool) {
	setReturnCallback(l.state, callback)
}

var (
	m         sync.Mutex
	callbacks = map[uintptr]func(string) bool{}
)

//export returntokenCallback
func returntokenCallback(statePtr unsafe.Pointer, token *C.char) bool {
	m.Lock()
	defer m.Unlock()

	if callback, ok := callbacks[uintptr(statePtr)]; ok {
		return callback(C.GoString(token))
	}

	return true
}

// setReturnCallback can be used to register a token callback for LLama. Pass in a nil callback to
// remove the callback.
func setReturnCallback(statePtr unsafe.Pointer, callback func(string) bool) {
	m.Lock()
	defer m.Unlock()

	if callback == nil {
		delete(callbacks, uintptr(statePtr))
	} else {
		callbacks[uintptr(statePtr)] = callback
	}
}
