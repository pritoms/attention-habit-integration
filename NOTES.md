# Attention Habit Integration

In this notebook, we will implement a set of algorithms for building something we would call **attention-habit**, which any typical pytorch model such as `transformers` could integrate as a layer.

## Outline

- Define attention computation methods
- Construct a new model called `AttentionHabit` that is parameterized by the node being implemented.
- The node is embedded in a $n \times k$ dimensional space, where $n$ refers to the size of the parent space, and $k$ is the size of the child space.
- The child space encodes the local embedding for the node being implemented.
- Finally, the objective is to setup the node management process such that any new input to global space or up to some definite scope reachable to the local data points of is sensitive to the node.

## Attention Computation Methods

- We will be using `transformers`.
- We will define our own attention computation methods that will be used in `AttentionHabit` to compute the attention weights.
- The inputs are the `query` and the `key` which is a sequence of vectors.
- Our methods will be:
  - `scaled_dot_product`: compute the dot product between query and each key, divide by $\sqrt{d_k}$ where $d_k$ is the dimensionality of each key vector.
  - `cosine_similarity`: compute the cosine similarity between query and each key, applied to a batch of queries and keys.
  - `multihead_attention`: implement a multihead attention computation method.
  - `multihead_attention_with_projections`: implement a multihead attention computation method with projections for query, key, value, and out.
    - This method is supposed to be called by the `AttentionHabit`.

## Attention Habit Layer

Here, we will implement a class called `AttentionHabit` that inherits from `torch.nn.Module`.

- The constructor of the class takes in the following arguments:
  - `parent_size`: the size of the parent space.
  - `child_size`: the size of the child space.
  - `attention_heads`: the number of attention heads.
  - `attention_dim`: the dimensionality of the attention layer.
- The forward method takes in two arguments:
  - `init_data`: a tensor that holds the initial data points for the layer, each data point is a vector in $n \times k$ dimensional space.
  - `input_data`: a tensor that holds the input data to be integrated with the initial data points, each data point is a vector in $n$ dimensional space.
- The output of the forward method is a tensor with shape $m \times n$ where $m$ is the number of data points in `input_data`.

The forward method will perform the following steps:

- embedd `input_data` into a tensor with shape $m \times d$ where $d$ is the dimensionality of each data point in the projected space.
- compute the `query`, `key`, and `value` tensors.
- compute the attention weights and the attention output.
- return the attention output.

## Attention Habit Integration

- We will be using a BERT model.
- The objective is to integrate the attention habit layer into the model and train it on a dataset.
