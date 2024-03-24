
class Dataset:
    def __init__(self, tables):
        self.tables = tables

    def filter(self, filter_lambda):
        for name, table in self.tables.items():
            self.set(name, filter_lambda(table))    
        return self

    def get_tables(self):
        return self.tables.items()
    
    def get_table(self, name):
        return self.tables[name]
    
    def get_base(self):
        return self.tables["base"]
    
    def get_depth_tables(self, depths):
        if type(depths) is not list:
            depths = [depths]
        depths = [str(i) for i in depths]

        for name, table in self.tables.items():
            if name == "base":
                continue
            if name[-1] in depths:
                yield name, table

    def set(self, name, table):
        self.tables[name] = table
    
    def delete(self, name):
        del self.tables[name]
