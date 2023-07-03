<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Open Flamingo Module

This repository contains the code supporting the Open Flamingo base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Open Flamingo](https://github.com/mlfoundations/open_flamingo) is an open-source implementation of the DeepMind Flamingo model.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [Open Flamingo Autodistill documentation](https://autodistill.github.io/autodistill/base_models/open-flamingo/).

## Installation

To use Open Flamingo with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-open-flamingo
```

## Quickstart

```python
from autodistill_open_flamingo import OpenFlamingo

# define an ontology to map class names to our OpenFlamingo prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = OpenFlamingo(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```


## License

This project is licensed under an MIT license.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!