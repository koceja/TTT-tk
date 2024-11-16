### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'ttt_linear_forward': {
        'source_files': {
            'h100': 'kernels/ttt_linear/ttt_linear_forward.cu'
        }
    },
    'ttt_mlp_forward': {
        'source_files': {
            'h100': 'kernels/ttt_mlp/ttt_mlp_forward.cu'
        }
    },
    'ttt_tp': {
        'source_files': {
            'h100': 'kernels/ttt_tp/ttt_tp.cu'
        }
    },
    'ttt_mlp_forward_tp': {
        'source_files': {
            'h100': 'kernels/ttt/ttt.cu'
        }
    },
    'attn': {
        'source_files': {
            'h100': 'kernels/attn/h100/h100.cu' # define these source files for each GPU target desired.
        }
    },
    'hedgehog': {
        'source_files': {
            'h100': 'kernels/hedgehog/hh.cu'
        }
    },
    'based': {
        'source_files': {
            'h100': [
                'kernels/based/lin_attn_h100.cu',
            ]
        }
    },
    'cylon': {
        'source_files': {
            'h100': 'kernels/cylon/cylon.cu'
        }
    },
    'flux': {
        'source_files': {
            'h100': [
                'kernels/flux/flux_gate.cu',
                'kernels/flux/flux_gelu.cu'
            ]
        }
    },
    'fftconv': {
        'source_files': {
            'h100': 'kernels/fftconv/pc/pc.cu'
        }
    },
    'fused_rotary': {
        'source_files': {
            'h100': 'kernels/rotary/pc.cu'
        }
    },
    'fused_layernorm': {
        'source_files': {
            'h100': 'kernels/layernorm/non_pc/layer_norm.cu'
        }
    },
    'mamba2': {
        'source_files': {
            'h100': 'kernels/mamba2/pc.cu'
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
kernels = ['ttt_mlp_forward_tp']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = 'h100'
