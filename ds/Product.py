
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
        with open(f"{path}/products.txt", "r") as file:
            s = file.read().split("\n")
            lines = Product.clean_lines(s)
            lines = Product.parse_products(lines)
            return [Product(line[0], line[1], line[2]) for line in lines]

    @staticmethod
    def to_string(products):
        for p in products:
            print("\n")
            print(f"name: {p.name}\nprice: {p.price}\nquantity: {p.quantity}")

    @staticmethod
    def clean_lines(lines):
        return [line.strip() for line in lines if line.strip()]

    @staticmethod
    def parse_products(lines):
        result = []
        for line in lines:
            tpl = line.split(",")
            if len(tpl) == 3:
                try:
                    result.append((tpl[0], float(tpl[1]), int(tpl[2])))
                except Exception as e:
                    raise e

        return result
