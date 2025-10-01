# YOSO

## Constraints

Time constraints will significantly impact the approach

- If the image needs to be rendered in near real time (for example, if a client uploads an image to the website to check if a garment suits them or not): fast solution is preferred
- If time is not a constraint (but mainly labor cost): We can opt for a solution that gives the best possible quality. => Since we focus on color correction, this is the right approach.

## Tasks

- Construct the dataset (data augmentation?)
- Inference
- Choose the metrics
- Model
- Deployment
- Refactor the modules to namespaces

## Achievements

- Package the library
- Set up CI/CD

## Future improvements

- More precise mask

### Metrics

- Hedonic value
- Utilitarian value

## Data

- VITON-HD: <https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset>

## Discussion

Some of the required input for color correction should already be available from earlier step.

## Installation

## Contributing

## Installing development requirements

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv ~/venv yoso --python 3.10
source ~/venv/yoso/bin/activate
uv pip install -e .
```

## Running the tests

```bash
pytest tests
```

## Credits

This project reuses/modifies code from **[IDM-VTON](https://github.com/yisol/IDM-VTON)** by [Yisol](https://github.com/yisol), licensed under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**.

## License

The codes and checkpoints in this repository are under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
