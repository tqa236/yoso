# YOSO

## Constraints

Time constraints will significantly impact the approach

- If the image needs to be rendered in near real time (for example, if a client uploads an image to the website to check if a garment suits them or not): fast solution is preferred
- If time is not a constraint (but mainly labor cost): We can opt for a solution that gives the best possible quality.

## Tasks

- Construct the dataset
- Lit review
  - https://arxiv.org/abs/2211.09800
  - https://arxiv.org/pdf/1911.02685
- Model
- Deployment

## Data

- VITON-HD: <https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset>

## Installation

## Contributing

### Installing development requirements

```bash
uv ~/venv yoso --python 3.13
source ~/venv/yoso/bin/activate
uv pip install -r requirements/requirements.txt
```

### Running the tests

```bash
pytest tests
```
