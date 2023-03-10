from .operators_dict import OPERATOR_DICT

class Operator:
    def __init__(self, name, params=None, zero_invariant=False, type="", shape_dependent=False, op_dict=None):
        self.name = name.lower()
        self.zero_invariant = zero_invariant
        self.type = type
        self.shape_dependent = shape_dependent
        self.params = params if params else {}
        if op_dict is not None:
            self.zero_invariant = True if op_dict['Zero-Invariant'] == 'TRUE' else False
            self.type = op_dict['Type']
            self.shape_dependent = True if op_dict['Shape-Dependent'] == 'TRUE' else False

    def __repr__(self):
        return self.full_info()

    def full_info(self):
        return "Name: " + self.name + ", zero_invariant: " + str(self.zero_invariant) + ", type: " + self.type + \
              ", shape_dependent: " + str(self.shape_dependent)

    def __eq__(self, op_name):
        return self.name == op_name

    def __add__(self, *other_ops):
        new_name = self.name
        new_zero_invariant = self.zero_invariant
        new_type = self.type
        new_shape_dependent = self.shape_dependent
        for op in other_ops:
            new_name += '_'  + op.name
            new_zero_invariant &= op.zero_invariant
            new_type += '_' + op.type
            new_shape_dependent &= op.shape_dependent
        return Operator(name=new_name, zero_invariant=new_zero_invariant, type=new_type, shape_dependent=new_shape_dependent)

OP_DICT = {}
for op_name in OPERATOR_DICT:
    op_dict = OPERATOR_DICT[op_name]
    op_name = op_name.lower()
    op = Operator(name=op_name, op_dict=op_dict)
    OP_DICT[op_name] = op
