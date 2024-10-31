# from whisperspeech.vq_stoks import RQBottleneckTransformer

# # Initialize
# tokenizer = RQBottleneckTransformer()

# # From audio file
# tokens = tokenizer.encode_audio("harvard.wav")

# import pdb; pdb.set_trace()

from whisperspeech.vq_stoks import make_model, Tunables

# Create a model with specific configuration
tokenizer = make_model('medium-2d-512c-dim64')

# From audio file
tokens = tokenizer.encode_audio("harvard.wav")

codebook = tokenizer.rq.layers[0]._codebook.embed[0]  # Get the codebook
vectors = codebook[tokens]

import pdb; pdb.set_trace()