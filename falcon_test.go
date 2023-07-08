package ggllm_test

import (
	. "github.com/mudler/go-ggllm.cpp"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Falcon GGLLM binding", func() {
	Context("Declaration", func() {
		It("fails with no model", func() {
			model, err := New("not-existing")
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})
	})
})
