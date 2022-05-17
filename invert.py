# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
from tqdm import tqdm
import numpy as np

from invert_utils.inverter import StyleGANInverter
from invert_utils.visualizer import resize_image

gpu_id = "0"
model_name = "styleganinv_ffhq256"
learning_rate = 0.01
num_iterations = 100
num_results = 5
loss_weight_feat = 5e-5
loss_weight_enc = 2.0
VIZ_SIZE = 256

def inversion(image):
  """Main function."""
  os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
  inverter = StyleGANInverter(
      model_name,
      learning_rate=learning_rate,
      iteration=num_iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=loss_weight_feat,
      regularization_loss_weight=loss_weight_enc)
  image_size = inverter.G.resolution

  image_list = []
  image_list.append(image)

  # Invert images.
  latent_codes = []
  for img_idx in tqdm(range(len(image_list)), leave=False):
    image = resize_image(image_list[img_idx], (image_size, image_size))
    code, viz_results = inverter.easy_invert(image, num_viz=num_results)
    latent_codes.append(code)

  return np.asarray(latent_codes)[0]


if __name__ == '__main__':
  main()
