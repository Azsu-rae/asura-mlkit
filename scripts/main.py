
from utils import clean_lines, parse_products
from utils import Product

products = Product.read_sample()
print([product.total_value() for product in products])
