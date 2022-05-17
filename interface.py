# python 3.7
"""Demo."""
import base64
from io import BytesIO
import requests
import numpy as np
import torch
import streamlit as st
import SessionState

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight
from PIL import Image
import torchvision.transforms as T
from invert import inversion

url = "https://t06twtw4n1.execute-api.us-west-1.amazonaws.com/dev/transform"
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(model_name):
    """Gets model by name."""
    return load_generator(model_name)


@st.cache(allow_output_mutation=True, show_spinner=False)
def factorize_model(model, layer_idx):
    """Factorizes semantics from target layers of the given model."""
    return factorize_weight(model, layer_idx)


def sample(model, gan_type, num=1):
    """Samples latent codes."""
    codes = torch.randn(num, model.z_space_dim)
    if gan_type == 'pggan':
        codes = model.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = model.net.mapping(codes)
        codes = model.net.truncation(codes)
    elif gan_type == 'stylegan2':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.5,
                                 trunc_layers=18)
    codes = codes.detach().cpu().numpy()
    return codes


@st.cache(allow_output_mutation=True, show_spinner=False)
def synthesize(model, gan_type, code):
    """Synthesizes an image with the give code."""
    if gan_type == 'pggan':
        image = model(to_tensor(code))['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        image = model.net.synthesis(to_tensor(code))[0]
    image = postprocess(image)[0]
    return image


def main():
    """Main function (loop for StreamLit)."""
    st.title('Closed-Form Factorization of Latent Semantics in GANs')
    st.sidebar.title('Options')

    model_name = st.sidebar.selectbox(
        'Model to Interpret',
        ["Hosoda", "Hayao", "Shinkai", "Paprika"])
    model_names = ["Hosoda", "Hayao", "Shinkai", "Paprika"]
    load_size = st.sidebar.slider("Set image size", 100, 800, 300, 20)

    # Settings
    model = get_model("stylegan_ffhq256")
    gan_type = parse_gan_type(model)
    layer_idx = st.sidebar.selectbox(
        'Layers to Interpret',
        ['all', '0-1', '2-5', '6-13'])
    layers, boundaries, eigen_values = factorize_model(model, layer_idx)

    num_semantics = st.sidebar.number_input(
        'Number of semantics', value=10, min_value=0, max_value=None, step=1)
    steps = {sem_idx: 0 for sem_idx in range(num_semantics)}
    max_step = 2.0
    for sem_idx in steps:
        eigen_value = eigen_values[sem_idx]
        steps[sem_idx] = st.sidebar.slider(
            f'Semantic {sem_idx:03d} (eigen value: {eigen_value:.3f})',
            value=0.0,
            min_value=-max_step,
            max_value=max_step,
            step=0.04 * max_step)

    button_placeholder = st.empty()
    image_placeholder = st.empty()

    base_codes = sample(model, gan_type)
    uploaded_image = st.sidebar.file_uploader(
        "Upload image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        state = SessionState.get(model_name="stylegan_ffhq256",
                                 code_idx=0,
                                 codes=base_codes[0:1])
        if state.model_name != model_name:
            pil_image = Image.open(uploaded_image)    
            image = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
            data = {
                "image": image,
                # TODO: More styles
                "model_id": model_names.index(model_name),
                "load_size": load_size,
            }
            response = requests.post(url, json=data)
            image = response.json()["output"]
            image = image[image.find(",") + 1:]
            dec = base64.b64decode(image + "===")
            binary_output = BytesIO(dec)
            cartoonified_img = np.asarray(Image.open(binary_output))
            base_codes = inversion(cartoonified_img)
            state.model_name = model_name
            state.code_idx = 0
            state.codes = base_codes[0:1]
        if button_placeholder.button('Random', key=0):
            state.code_idx += 1
            if state.code_idx < base_codes.shape[0]:
                state.codes = base_codes[state.code_idx][np.newaxis]
            else:
                state.codes = sample(model, gan_type)

        code = state.codes.copy()
        for sem_idx, step in steps.items():
            code[:, layers, :] += boundaries[sem_idx:sem_idx + 1] * step
        image = synthesize(model, gan_type, code)
        image_placeholder.image(image / 255.0)


if __name__ == '__main__':
    main()
