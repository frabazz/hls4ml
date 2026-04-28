import glob
import os
import stat
import tarfile
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile, copytree, rmtree
import logging

import numpy as np
import yaml

from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class BambuWriter(Writer):
    def print_array_to_cpp(self, var, odir, namespace=None, write_txt_file=True):
        h_file = open(f'{odir}/firmware/weights/{var.name}.h', 'w')
        if write_txt_file:
            txt_file = open(f'{odir}/firmware/weights/{var.name}.txt', 'w')

        # meta data
        h_file.write(f'//Numpy array shape {var.shape}\n')
        h_file.write(f'//Min {np.min(var.min):.12f}\n')
        h_file.write(f'//Max {np.max(var.max):.12f}\n')
        h_file.write(f'//Number of zeros {var.nzeros}\n')
        h_file.write('\n')

        h_file.write(f'#ifndef {var.name.upper()}_H_\n')
        h_file.write(f'#define {var.name.upper()}_H_\n')
        h_file.write('\n')

        if namespace is not None:
            h_file.write(f'namespace {namespace} {{\n\n')

        if write_txt_file:
            h_file.write('#ifndef __BAMBU__\n')
            h_file.write(var.definition_cpp() + ';\n')
            h_file.write('#else\n')

        h_file.write(var.definition_cpp() + ' = {')

        sep = ''
        for x in var:
            h_file.write(sep + x)
            if write_txt_file:
                txt_file.write(sep + x)
            sep = ', '
        h_file.write('};\n\n')

        if write_txt_file:
            h_file.write('#endif\n')
            txt_file.close()

        if namespace is not None:
            h_file.write('}\n\n')

        h_file.write('\n#endif\n')
        h_file.close()

    def write_project_dir(self, model):
        if not os.path.isdir(f'{model.config.get_output_dir()}/firmware/weights'):
            os.makedirs(f'{model.config.get_output_dir()}/firmware/weights')

    @staticmethod
    def _make_array_pragma(variable):
        config = variable.pragma
        if type(config) is tuple:
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode in ['partition', 'reshape']:
            if typ == 'complete':
                template = '//#pragma HLS ARRAY_{mode} variable={name} {type} dim={dim}'
            else:
                template = '//#pragma HLS ARRAY_{mode} variable={name} {type} factor={factor} dim={dim}'

            return template.format(mode=mode.upper(), name=variable.name, type=typ, factor=factor, dim=0)

        elif mode == 'stream':
            return f'//#pragma HLS STREAM variable={variable.name} depth={depth}'

    def write_project_cpp(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/bambu/firmware/myproject.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str
                if len(model_brams) > 0:
                    newline += ',\n' + brams_str
                newline += '\n'

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''
                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''
                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                if model.config.get_writer_config()['WriteWeightsTxt']:
                    newline += '#ifndef __BAMBU__\n'
                    newline += '    static bool loaded_weights = false;\n'
                    newline += '    if (!loaded_weights) {\n'

                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            if w.weight_class == 'CompressedWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.nonzeros, w.name, w.name
                                    )
                                )
                            elif w.weight_class == 'ExponentWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.data_length, w.name, w.name
                                    )
                                )
                            else:
                                newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                    w.type.name, w.data_length, w.name, w.name
                                )

                    newline += '        loaded_weights = true;'
                    newline += '    }\n'
                    newline += '#endif'

            elif '// hls-fpga-machine-learning insert IO' in line:
                newline = line
                all_inputs = [i.name for i in model_inputs]
                all_outputs = [o.name for o in model_outputs]
                all_brams = [b.name for b in model_brams]
                io_type = model.config.get_config_value('IOType')

                pipeline_style = model.config.pipeline_style
                pipeline_ii = model.config.pipeline_ii
                pipeline_pragma = indent + f'//#pragma HLS {pipeline_style.upper()}'
                if pipeline_style == 'pipeline' and pipeline_ii is not None:
                    pipeline_pragma += f' II={pipeline_ii}\n'
                else:
                    pipeline_pragma += '\n'

                if io_type == 'io_parallel':
                    for i in model_inputs:
                        newline += indent + self._make_array_pragma(i) + '\n'
                    for o in model_outputs:
                        newline += indent + self._make_array_pragma(o) + '\n'
                    newline += indent + '#pragma HLS interface mode=valid port={},{} \n'.format(
                        ','.join(all_inputs), ','.join(all_outputs)
                    )
                    newline += pipeline_pragma

                if io_type == 'io_stream':
                    newline += indent + '#pragma HLS interface mode=axis port={},{} \n'.format(
                        ','.join(all_inputs), ','.join(all_outputs)
                    )
                    if all_brams:
                        newline += indent + '//#pragma HLS INTERFACE bram port={} \n'.format(','.join(all_brams))
                    newline += pipeline_pragma

            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in model_inputs and var not in model_outputs:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                newline += '    ' + def_cpp + ';\n'
                                if var.pragma:
                                    newline += '    ' + self._make_array_pragma(var) + '\n\n'
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func:
                        if not isinstance(func, (list, set)):
                            func = [func]
                        if len(func) == 1:
                            newline += '    ' + func[0] + ' // ' + layer.name + '\n'
                        else:
                            newline += '    // ' + layer.name + '\n'
                            for line in func:
                                newline += '    ' + line + '\n'
                        if model.config.trace_output and layer.get_attr('trace', False):
                            vars = layer.get_variables()
                            newline += '#ifndef __BAMBU__\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                    var.type.name, var.name, layer.name, var.size_cpp()
                                )
                            newline += '#endif\n'
                        newline += '\n'

            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_project_header(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/bambu/firmware/myproject.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.h', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))

            elif 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str
                if len(model_brams) > 0:
                    newline += ',\n' + brams_str
                newline += '\n'

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''
                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''
                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif '// hls-fpga-machine-learning insert emulator-defines' in line:
                newline = line

                if model.config.get_writer_config().get('WriteEmulationConstants', False):
                    brams_def_str = ', '.join([b.definition_cpp(as_reference=False) for b in model_brams])
                    brams_call_str = ', '.join([b.name for b in model_brams])

                    if model.config.get_config_value('IOType') == 'io_stream':
                        input_call_str = ', '.join([f'std::get<{n}>(inputs)' for n in range(len(model_inputs))])
                        output_call_str = ', '.join([f'std::get<{n}>(outputs)' for n in range(len(model_outputs))])
                    else:
                        input_call_str = ', '.join([f'std::get<{n}>(inputs).data()' for n in range(len(model_inputs))])
                        output_call_str = ', '.join([f'std::get<{n}>(outputs).data()' for n in range(len(model_outputs))])

                    newline += (
                        f'\ninline void {model.config.get_project_name()}_emulator('
                        'inputs_t& inputs, outputs_t& outputs'
                    )
                    if len(model_brams) > 0:
                        newline += ',\n' + brams_def_str
                    newline += ') {\n'
                    newline += indent + model.config.get_project_name() + '(\n'
                    newline += indent + indent + input_call_str + ',\n'
                    newline += indent + indent + output_call_str
                    if len(model_brams) > 0:
                        newline += ',\n' + indent + indent + brams_call_str
                    newline += '\n' + indent + ');\n}\n'

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_defines(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(file
