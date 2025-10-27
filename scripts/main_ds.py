
from ds import Product

products = Product.read_sample()
print([product.total_value() for product in products])
