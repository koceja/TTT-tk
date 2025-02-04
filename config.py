### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'ttt': {
        'source_files': {
            'h100': 'kernels/ttt/ttt.cu'
        }
    },
    'ttt_backward': {
        'source_files': {
            'h100': 'kernels/ttt_backward/ttt.cu'
        }
    },
    'attn': {
        'source_files': {
            'h100': 'kernels/attn/h100/h100.cu' # define these source files for each GPU target desired.
        }
    },
    'fused_layernorm': {
        'source_files': {
            'h100': 'kernels/layernorm/non_pc/layer_norm.cu'
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
kernels = ['ttt', 'ttt_backward']
# kernels = ['ttt_backward']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = 'h100'
