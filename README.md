# tf-advanced-techniques

Advanced TensorFlow techniques — custom model components, computer vision beyond classification, distributed training strategies, and generative deep learning.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Overview

This repository covers the advanced tier of the TensorFlow specialization: moving beyond `model.fit()` to build fully custom training loops, novel architectures, and generative models. Implementations use the Keras functional API and subclassing API throughout.

---

## Repository Structure

```
tf-advanced-techniques/
├── custom-models-layers-loss-functions/
│   ├── custom_loss_function.ipynb                        # Huber loss, contrastive loss from scratch
│   ├── multiple_output_models_keras_functional_api.ipynb # Multi-head model with shared backbone
│   ├── quadratic_layer.ipynb                             # Custom Layer subclass with trainable params
│   └── vgg_network.ipynb                                 # VGG-16 reimplemented via functional API
├── Custom and Distributed Training with TensorFlow/
│   ├── C2W1_Assignment.ipynb                             # Custom training loop with GradientTape
│   └── C2W2_Assignment.ipynb                             # Distributed strategies (MirroredStrategy)
├── Advanced Computer Vision with TensorFlow/
│   └── (GradCAM, object detection, image segmentation)
└── Generative Deep Learning with TensorFlow/
    └── (VAE, GAN, neural style transfer)
```

---

## Topics Covered

| Module | Techniques |
|--------|-----------|
| **Custom Components** | Custom loss functions, custom layers (`Layer` subclass), multi-output models |
| **Custom Training** | `tf.GradientTape`, manual gradient application, custom training loops |
| **Distributed Training** | `MirroredStrategy`, multi-GPU training patterns |
| **Advanced CV** | GradCAM saliency maps, object detection, semantic segmentation |
| **Generative Models** | Variational Autoencoders (VAE), GANs, neural style transfer |

---

## Getting Started

### Prerequisites

- Python 3.10+
- TensorFlow 2.x
- GPU recommended for generative model notebooks

### Installation

```bash
git clone https://github.com/Rifqidits/tf-advanced-techniques.git
cd tf-advanced-techniques
pip install -r requirements.txt
jupyter notebook
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
