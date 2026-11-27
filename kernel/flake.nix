{
  description = "Sparton SPLADE projection and pooling";

  inputs.kernel-builder.url = "github:huggingface/kernels/7f96de84460f398968560c9182ee6f2180918814";

  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
