# Compute FID value...
wget --no-check-certificate 'https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth'

# Compute content distance...
wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth -O alexnet-owt-7be5be79.pth
mkdir -p ~/.cache/torch/hub/checkpoints/
mv alexnet-owt-7be5be79.pth ~/.cache/torch/hub/checkpoints/

# Compute CFSD value...
wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth --no-check-certificate
mkdir -p ~/.cache/torch/hub/checkpoints/
mv vgg19-dcbb9e9d.pth ~/.cache/torch/hub/checkpoints/
