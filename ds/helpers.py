
def clean_lines(lines):
    return [line.strip() for line in lines if line.strip()]

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
