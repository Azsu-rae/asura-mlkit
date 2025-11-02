
from ds import Product
import matplotlib.pyplot as plt

products = Product.read_sample()

names = [p.name for p in products]
values = [p.total_value() for p in products]
print(names)
print(values)

plt.bar(names, values)
plt.xlabel('names')
plt.ylabel('total values')
plt.show()
