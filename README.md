```markdown
# MicroGrad-NN

Minimal micrograd-style autograd engine + neural network library with Graphviz visualization.

## Structure
```

engine.py   # Value class, autograd, graph tracing, draw_dot()
nn.py       # Neuron, Layer, MLP
demo.py     # scalar tests, activations, NN, XOR training

````

## Install
```bash
pip install graphviz
````

### System dependency

**Windows**

```
https://graphviz.org/download/
Add: C:\Program Files\Graphviz\bin  → PATH
```

**Linux**

```bash
sudo apt install graphviz
```

## Run demo

```bash
python demo.py
```

## Jupyter visualization

```python
from IPython.display import display
from engine import Value, draw_dot
from nn import MLP

model = MLP(3, [4,4,1])
x = [0.5, -1.0, 2.0]

y = model(x)
y.grad = 1.0
y.backward()

display(draw_dot(y))
```

## Output

* Full computation graph (autograd DAG)
* Values + gradients
* Ops: `+`, `*`, `tanh`, `relu`, `exp`, `**`

## Purpose

Educational reference implementation of:

* Reverse-mode autodiff
* Computational graphs
* Backpropagation
* Minimal neural networks
* Symbolic graph visualization

```
```
