
from .helpers import clean_lines, parse_products

from pathlib import Path
path = Path.home() / "Data"

class Product:

    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

    def total_value(self):
        return self.price * self.quantity

    @staticmethod
    def read_sample():
        lines = ""
        with open(f"{path}/products.txt", "r") as file:
            print(file.read())
            lines = clean_lines(file.read())
            lines = parse_products(lines)
            return [Product(line[0], line[1], line[2]) for line in lines]
