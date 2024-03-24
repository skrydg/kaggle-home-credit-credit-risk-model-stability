import copy

class ColumnsInfo:
    def __init__(self, raw_tables_info=None):
        self.raw_tables_info = raw_tables_info
        self.labels = {}
        self.ancestors = {}
        self.column_to_table_name = {}
    
    def set_raw_tables_info(self, raw_tables_info):
        self.raw_tables_info = raw_tables_info
    
    def get_raw_tables_info(self):
        return self.raw_tables_info

    def get_labels(self, column):
        if column not in self.labels:
            self.labels[column] = set()
        return copy.deepcopy(self.labels[column])
    
    def delete_label(self, column, label):
        assert(column in self.labels)
        self.labels[column].delete(label)
        
    def add_label(self, column, label):
        self.add_labels(column, set([label]))

    def add_labels(self, column, labels):
        if column not in self.labels:
            self.labels[column] = set()
        self.labels[column] = self.labels[column] | labels

    def get_columns_with_label(self, label):
        return [column for column, labels in self.labels.items() if label in labels]
    
    def has_ancestor(self, column):
        return column in self.ancestors
    
    def get_ancestor(self, column):
        return self.ancestors[column]
    
    def set_ancestor(self, column, ancestor_column):
        self.ancestors[column] = ancestor_column

    def get_table_name(self, column):
        return self.column_to_table_name[column]
    
    def set_table_name(self, column, table_name):
        self.column_to_table_name[column] = table_name