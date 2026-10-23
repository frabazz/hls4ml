from hls4ml.backends.fpga.fpga_types import APTypeConverter, HLSTypeConverter, StaticWeightVariableConverter
from hls4ml.backends.bambu.bambu_types import (
    BambuArrayVariableConverter,
    BambuInplaceArrayVariableConverter,
    BambuInplaceStreamVariableConverter,
    BambuStreamVariableConverter,
)
from hls4ml.model.layers import Conv1D, Conv2D, Dense, DepthwiseConv1D, DepthwiseConv2D
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.types import InplaceTensorVariable


class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.type_converter = HLSTypeConverter(precision_converter=APTypeConverter())
        self.array_var_converter = BambuArrayVariableConverter(type_converter=self.type_converter)
        self.inplace_array_var_converter = BambuInplaceArrayVariableConverter(type_converter=self.type_converter)
        self.stream_var_converter = BambuStreamVariableConverter(type_converter=self.type_converter)
        self.inplace_stream_var_converter = BambuInplaceStreamVariableConverter(type_converter=self.type_converter)
        self.weight_var_converter = StaticWeightVariableConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        io_type = node.model.config.get_config_value('IOType')

        for out_name, var in node.variables.items():
            if io_type == 'io_stream':
                if isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_stream_var_converter.convert(var)
                else:
                    new_var = self.stream_var_converter.convert(var)
            elif io_type == 'io_serial':
                new_var = self.array_var_converter.convert(var, pragma='stream')
            elif io_type == 'io_parallel':
                if out_name in node.model.inputs:
                    # NOTE this needs to be changed to partition
                    new_var = self.array_var_converter.convert(var, pragma='reshape')
                elif isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_array_var_converter.convert(var, pragma='')
                else:
                    new_var = self.array_var_converter.convert(var, pragma='partition')
            else:
                raise Exception(f'Unknown IOType {io_type} in {node.name} ({node.__class__.__name__})')

            node.set_attr(out_name, new_var)

        for w_name, weight in node.weights.items():
            new_weight = self.weight_var_converter.convert(weight)
            block_factor = self._weight_block_factor(node, new_weight) if w_name == 'weight' else None
            if block_factor is not None:
                new_weight.pragma = ('reshape', 'block', block_factor)
            node.set_attr(w_name, new_weight)

        for t_name, type in node.types.items():
            new_type = self.type_converter.convert(type)
            node.set_attr(t_name, new_type)

    @staticmethod
    def _weight_block_factor(node, weight):
        # The Resource kernels reshape the weights (ARRAY_RESHAPE block factor=block_factor, commented in the Bambu
        # templates). The writer emits it as the equivalent block partition, with the factor as a literal since Bambu
        # does not evaluate CONFIG_T expressions in pragmas.
        if not isinstance(node, (Dense, Conv1D, Conv2D)) or isinstance(node, (DepthwiseConv1D, DepthwiseConv2D)):
            return None
        reuse_factor = node.get_attr('reuse_factor', 1)
        if reuse_factor <= 1 or not node.model.config.is_resource_strategy(node):
            return None
        # DIV_ROUNDUP(n_in * n_out, reuse_factor) as in the templates: n_in * n_out is the number of weights
        return -(-weight.data_length // reuse_factor)
