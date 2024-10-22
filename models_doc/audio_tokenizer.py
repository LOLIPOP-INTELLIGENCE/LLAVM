import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from whisperspeech.vq_stoks import Tunables
from models.residual_vq import ResidualVQ
import math

class RQBottleneckTransformer(nn.Module):
    def __init__(self, vq_codes = 512, q_depth = 12, depth = 1, n_head = 2,\
                 head_width = 64, ffn_mult = 4, codebook_dim =2, threshold_ema_dead_code= 2,\
                 use_cosine_sim = False, kl_loss_mul =1, downsample = 1, no_quantize = False,\
                 whisper_model_name = 'tiny.en', tunables = Tunables()):
        super().__init__()
        width = n_head * head_width
        self.width = width
        self.base_width = 3 * head_width
        self.vq_codes = vq_codes
        self.tunables = tunables
        self.stoks_len = 1500 // downsample
        self.stoks_per_sec=  self.stoks_len //30 #Whisper takes in batches of 30 seconds, so divide by 30
        self.no_quantize = no_quantize

        qk_scale = self.tunables.query_mult * 8 / math.sqrt(head_width) # This is q_mult/sqrt(d)

        self.kl_loss_mul = kl_loss_mul

        if no_quantize:
        #Gets whisper baseline into W&B easily, skips all training
            self.fake_parameter = nn.Parameter(torch.tensor(0.001))
        else:
            n_mlp = width * ffn_mult
            self.mlp = nn.Sequential(
                nn.Linear(width, n_mlp),
                nn.GELU(),
                nn.Linear(n_mlp, width)
            )

            self.mlp_ln = nn.LayerNorm(width)

            if tunables.downsample_conv:
                self.downsample_conv = nn.Conv1d(width, width, kernel_size = 3, stride = downsample, padding = 1 )
            else:
                self.downsample_conv = None
                
            if tunables.mask_embs: 
                vq_codes = vq_codes +1 #Because autoregressively decoding next mask
            self.rq = ResidualVQ(dim = width, 
                                 codebook_size = vq_codes,
                                 decay = tunables.codebook_decay,
                                 commitment_weight = 1., #Commitment loss weight,
                                 threshold_ema_dead_code = threshold_ema_dead_code,
                                 use_cosine_sim = use_cosine_sim,
                                 codebook_dim = codebook_dim,
                                 num_quantizers = 1,
                                 )
            self.positional_embedding = nn.Embedding(1500, self.stoks_len)
            
            self._out_blocks = nn.Sequential(*[ResudualAttentionBlock(width, n_head, qk_scale = qk_scale, ffn_mult = ffn_mult, rope =tunables.rope) for _ in range(depth)])
            self.ln_post = nn.LayerNorm(width)
        self.positions = torch.arangr(0,1500,dtype = torch.long)
        self.ce_lossf = nn.CrossEntropyLoss(ignore_index = -100) #Padding is -100 so ignore This
        self.kl_lossf = nn.KLDivLoss(reduction = 'batchmean')
        self.whmodel=  None
        self.apply(self.init_transformer)
        self.register_buffer('val_true', torch.zeros(1))
        self.register_buffer('val_total', torch.zeros(1))


    def setup(self, device):
        self.ensure_whisper(device)

    def init_transformer(self, m):
        if isinstance(m, LinearHead):
            m.no_weight_decay = True
            torch.nn.init.constant_(m.weight, 0)
        elif isinstance(m, QueryHead):
            m.lr_scale =1/(m.weight.shape[1] / self.base_width) #Scales learning rate inversely with layer width
            torch.nn.init.constant_(m.weight, 0)
        elif isinstance(m, nn.Embedding):
            m.no_weight_decay = True
            m.lr_scale = self.tunables.embeddings_lr_scale
            std = self.tunables.embeddings_std
        elif isinstance(m, nn.Linear):
            m.lr_scale = 1/(m.weight.shape[1] / self.base_width)
            std = self.tunables.init_std / m.weight.shape[1]
            torch.nn.init.trunc_normal_(m.weight, std=  std, a = -3*std, b = 3*std)
            if m.bias is not None:
                torch.nn.init.trunc_normal_(m.bias, std = std, a = -3*std, b = 3*std)
            elif isinstance(m, nn.LayerNorm):
                m.no_weight_decay = True
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1)
    
    @property
    def device(self):  
        return next(self.parameters()).device

    ###### Training ########
    def log_mel_spectrogram(self, samples):
        # Creates the log mel spectrogram using whisper
        return whisper.log_mel_spectrogram(samples, 128 if self.whisper_model_name == 'large-v3' else 80)

    @torch.no_grad()
    def extract_teacher(self, samples, input_toks, output_toks):
        embs = self.whmodel[0].encoder(self.log_mel_spectrogram(samples))
        teacher_logits=  self.whmodel[0].decoder(input_toks, embs)
        #Set teacher logits to 0 for padding positions so KLDivLoss ignores them 
        teacher_logits[output_tokens == -100] = 0
        return embs, teacher_logits

    def downsample_embeddings(self, x):
        if self.downsample_conv is not None:
            return x[:,::self.downsample] + self.downsample_conv(x.transpose(-1,-2)).transpose(-2,-1)
            # We transpose in the conv due to channels ordering in the shape
            # Here we downsample by stride and add the conv -> Why?

        elif self.tunables.downsample_mean: #Use mean to downsample instead
            bs, slen, depth = x.shape
            return x.reshape(bs,slen//self.downsample, self.downsample, depth).mean(-2)

        else:
            return x[:,::self.downsample]


    def out_blocks(self, x):
        #Shortens forward by having a helper function
        for l in self._out_blocks:
            x = l(x, self.positions)
            return x
    
    def forward(self, samples, mask, input_toks, output_toks):
        embs, teacher_logits = self.extract_teacher(samples, input_toks, output_toks)
        # This extracts teacher logits for reference later
        if not self.no_quantize:
            embs = self.downsample_embeddings(embs)
            x = x+self.mlp(self.mlp_ln(x))
            # VQ bottleneck
            quantized, self.indices, self.commit_loss = self.rq(x)
            self.commit_loss = self.commit_loss.mean()
            x = quantized.repeat_interleave(self.downsample, -2)
            project_out = getattr(self.rq, 'project_out', None) or self.rq.layers[0].project_out
            if self.tunables.mask_embs:
                x[~mask] = project_out(self.rq.layers[0]._codebook.embed[0, self.vq_codes])
                x = x + self.positional_embeding(self.positions.to(x.device))
                x = self.ln_post(self.out_blocks(x))
            



    

                

