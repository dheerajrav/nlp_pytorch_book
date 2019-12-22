import torch

def describe(x, description=None):
    """
    Prints the type, shape, and values of a Tensor
    """
    print(description)
    print("Type: {}".format(x.type()))
    print("Shape/Size: {}".format(x.shape))
    print("Values: {}".format(x))
    print("")

x = torch.rand(2, 3)
y = torch.rand(2, 3)
z = torch.randn(3, 2)

# Adding two tensors using arithmatic operator

describe(x + x, "Two tensor addition")

# Adding two tensors using .add

describe(torch.add(x, x), "Two tensor addition using add function")

# Subtracting two tensors arithmatic operator

describe(x - y, "Two tensor subtraction")

# Subtracting two tensors using sub function

describe(torch.sub(x, y), "Two tensor subtraction using sub function")

# Elementwise multiplication

describe(torch.mul(x, y), "Two tensor element-wise multiplication")

# Matrix multiplication

describe(torch.matmul(x, z), "Two tensor matrix multiplication")

# Multiplication using arithmatic operator

describe(x * y, "Two tensor multiplication using arithmatic operator")

# Changing the dimensions of a tensor

example_1 = torch.arange(6)
describe(example_1, "1, 6 dim tensor")

# Now I change it's shape to 2, 3

example_1a = example_1.view(2, 3)
describe(example_1a, "1, 6 -> 2, 3 dim changed tensor")

# Summing rows

example_1b = torch.sum(example_1a, dim=0)

describe(example_1b, "2, 3 dim tensor row sum")

# summing columns

example_1c = torch.sum(example_1a, dim=1)
describe(example_1c, "2, 3 dim tensor column sum")

# Transpose - torch.transpose(input, dim0, dim1)
# dim0 -> first dimension to be transposed
# dim1 -> second dimension to be transposed

example_1d = torch.transpose(example_1a, 0, 1)

describe(example_1d, "First and second dimension are transposed")

# Slicing

describe(example_1a[:, :2], "Slicing: All rows and first two columns")

describe(example_1a[:, 1:2], "Slicing: All rows and first column")

# indexing

describe(example_1a[1, 2], "Indexing: 2nd row - 3rd element")

# Non contiguous index select

indices = torch.LongTensor([0, 2])
describe(example_1a.index_select(dim=1, index=indices), "Non contiguous index select")

# Non contiguous row and column select

row_indices = torch.LongTensor([1, 3])
col_indices = torch.LongTensor([0, 2])
aa = torch.randn(4, 5)
describe(aa)
describe(aa[row_indices, col_indices], "Non Contiguous row and col indices, [1, 0], and [3, 2]")

# Concatenation

x = torch.arange(6).view(2, 3)
describe(torch.cat([x, x]), "Two tensors of (2, 3) size are concatenated across row dimension")

# Concatenation across columns

describe(torch.cat([x, x], dim=1), "Two tensors of (2, 3) size are concatenated across columns")

# Concatenation across a separate dimension

describe(torch.stack([x, x], dim=1), "Two tensors of (2, 3) size are stacked - concatenation across new dimension")

# Sliced data addtions

y = torch.ones(2, 3)
y[:, :1] += 2
describe(y, "adding 1 to 0th column of a ones tensor")
