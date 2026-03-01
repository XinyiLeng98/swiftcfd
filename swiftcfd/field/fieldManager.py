from swiftcfd.field.field import Field

class FieldManager():
    def __init__(self, mesh):
        self.mesh = mesh
        self.fields = {}

    def add_field(self, field_name):
        field = Field(self.mesh, field_name)
        field.old = Field(self.mesh, field_name + '_old')
        field.oldold = Field(self.mesh, field_name + '_oldold')
        field.oldoldold = Field(self.mesh, field_name + '_oldoldold')
        field.picard_old = Field(self.mesh, field_name + 'picard_old')
        self.fields[field.name] = field

    def get_field(self, name):
        return self.fields[name]

    def get_all_fields(self):
        return self.fields

    def update_solution(self):
        for _, field in self.fields.items():
            field.update_solution()

    def update_picard_solution(self):
        for _, field in self.fields.items():
            field.update_picard_solution()
