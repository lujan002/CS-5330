# Bruce A. Maxwell and Andy Zhao
# Spring 2026
# MNIST Transformer Class Template
#
#
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetConfig:

    def __init__(self,
                 name = 'vit_base',
                 dataset = 'mnist',
                 patch_size = 4,
                 stride = 2,
                 embed_dim = 48,
                 depth = 4,
                 num_heads = 8,
                 mlp_dim = 128,
                 dropout = 0.1,
                 use_cls_token = False,  
                 epochs = 15,
                 batch_size = 64,
                 lr = 1e-3,
                 weight_decay = 1e-4,
                 seed = 0,
                 optimizer = 'adamw',
                 device = 'mps',
                 ):


        # data set fixed attributes
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        # variable things
        self.name = name
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer
        self.device = device

        s = "Name,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,Dropout,CLS,Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
        s += "%s,%s,%d,%d,%d,%d,%d,%d,%0.2f,%s,%d,%d,%f,%f,%d,%s," % (
            self.name,
            self.dataset,
            self.patch_size,
            self.stride,
            self.embed_dim,
            self.depth,
            self.num_heads,
            self.mlp_dim,
            self.dropout,
            self.use_cls_token,
            self.epochs,
            self.batch_size,
            self.lr,
            self.weight_decay,
            self.seed,
            self.optimizer
            )
        self.config_string = s

        return

    

# Patch Embedding class
#
# A Vision Transformer splits the image into small patches, then turns
# each patch into a token embedding.
class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.

    Input:
        x of shape (B, C, H, W)

    Output:
        tokens of shape (B, N, D)

    where:
        B = batch size
        N = number of patches (tokens)
        D = embedding dimension
    """

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            stride: int,
            in_channels: int,
            embed_dim: int,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # - non-overlapping patches  (stride == patch_size)
        # - overlapping patches      (stride < patch_size)
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=stride,
        )

        # Each extracted patch is flattened into one vector
        self.patch_dim = in_channels * patch_size * patch_size

        # After flattening a patch, project it into embedding space.
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

        # Precompute how many patches will be produced for this image setup
        self.num_patches = self._compute_num_patches()

    def _compute_num_patches(self) -> int:
        """
        Compute how many patches are extracted in total.

        Number of positions along one spatial dimension:
            ((image_size - patch_size) // stride) + 1

        Since the image is square and the patch is square, total patches are:
            positions_per_dim * positions_per_dim
        """
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches and convert them to embeddings.

        Input:
            x shape = (B, C, H, W)

        Output:
            x shape = (B, N, D)
        """
        # Step 1: extract patches using nn.Unfold, the shape becomes (B, patch_dim, N)
        #   patch_dim = flattened size of one patch
        #   N = number of extracted patches
        x = self.unfold(x)

        # Step 2: move dimensions so each patch becomes one row/token.
        # Shape becomes: (B, N, patch_dim)
        x = x.transpose(1, 2)

        # Step 3: project each flattened patch into embedding space.
        # Shape becomes: (B, N, embed_dim)
        x = self.proj(x)

        return x




# The Transformer Network class
#
# network structure
#
# Patch embedding layer
# dropout
# Transformer layer (with dropout)
# Transformer layer (with dropout)
# Transformer layer (with dropout)
# Token averaging
# Linear layer w/GELU and dropout
# Fully connected output layer 10 nodes: softmax output
class NetTransformer(nn.Module):

    # the init method defines the layers of the network
    def __init__(self, config):

        # create all of the layers that have to store information
        super(NetTransformer, self).__init__()

        # make the patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        # how many tokens are there?
        num_tokens = self.patch_embed.num_patches
        print("Number of tokens: %d" % (num_tokens) )

        # does it use a classifier token or a global average token?
        self.use_cls_token = config.use_cls_token

        # if it uses a classifier node, create a source for the node
        if self.use_cls_token:
            self.cls_token = nn.Parameter( torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens+1
        else: # no CLS token
            self.cls_token = None
            total_tokens = num_tokens

        # need to include a learned positional embedding, one for each token
        self.pos_embed = nn.Parameter(
            torch.zeros( 1, total_tokens, config.embed_dim ) )
        self.pos_dropout = nn.Dropout( config.dropout ) # do I need this?

        # Use the Torch Transformer Encoder Layer
        # transformer layer includes
        # multi-head self attention
        # feedforward network
        # layer normalization
        # residual connections
        # dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = config.embed_dim,
            nhead = config.num_heads,
            dim_feedforward = config.mlp_dim,
            dropout = config.dropout,
            activation = 'gelu',
            batch_first = True,
            norm_first = True,
        )

        # Create a stack of transformer layers to build an encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = config.depth,
        )

        # final normalization layer prior to classification
        self.norm = nn.LayerNorm( config.embed_dim )

         # linear layer for classification
        self.classifier = nn.Sequential(
            nn.Linear( config.embed_dim, config.mlp_dim),
            nn.GELU(),
            #nn.Dropout(config.dropout),  # optional
            nn.Linear( config.mlp_dim, config.num_classes )
        )

        return

    def _init_parameters(self) -> None:
        """
        initialize special parameters
        - positional embedding
        - optional CLS token
        """
        nn.init.trunc_normal_(self.pos_embed, std = .02)

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std = 0.02 )

    

    # execute a forward pass
    """
    Input x: (B, 1, 28, 28)

    Output: logits: (B, num_classes)
    """
    def forward( self, x ):
        # This function needs to be completed.  Each empty comment is one command.
        # X is an instance of the NetTransformer class

        # execute the patch embedding layer
        x = self.patch_embed(x)

        # get the batch size (0 dimension of x)
        batch_size = x.size(0)

        # add the optional CLS token to the set 
        if self.use_cls_token:
            cls_token = self.cls_token.expand( batch_size, -1, -1 )
            x = torch.cat( [cls_token, x], dim = 1 )

        # add the learnable positional embedding to each token
        x = x + self.pos_embed
        # run the dropout layer right after the patch embedding
        x = self.pos_dropout(x)
        # run the transformer encoder
        x = self.encoder(x)

        # either pool the tokens or use the cls token (first token)
        if self.use_cls_token:
            x = x[:,0] # classify based on the cls token
        else:
            x = x.mean(dim=1) # classify using the mean token vector

        # final normalization of the token to classify
        x = self.norm(x)
        # call the classification MLP
        x = self.classifier(x)
        # return the softmax of the output layer
        return F.log_softmax( x, dim=1 ) # return softmax of the output layer


