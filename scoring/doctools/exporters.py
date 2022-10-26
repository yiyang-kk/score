import os

class Exporter(object):
    
    conf_nm = 'doctool-export-config.py'
    tpl_nm = 'doctool-template.tpl'
    ppc_nm = 'doctool-preprocessor.py'
    
    def export(self, nb_name):
        
        # documentation-config
        a="c = get_config() \nc.NbConvertApp.notebooks = ['" + nb_name + "'] \nc.NbConvertApp.export_format = 'html' \nc.Exporter.preprocessors = ['doctool-preprocessor.ExportLabeled'] \n" 
        
        b="{%- extends 'full.tpl' -%} \n{% block input_group %} \n{%- endblock input_group %}"
        
        c="from nbconvert.preprocessors import Preprocessor \n\
class ExportLabeled(Preprocessor): \n\
	def preprocess(self, notebook, resources):\n\
		notebook.cells = [cell for cell in notebook.cells if 'documentation' in cell.metadata]\n\
		return notebook, resources\n"
        
        with open(self.conf_nm, 'w') as outfile:
            outfile.write(a)
            
        with open(self.tpl_nm, 'w') as outfile:
            outfile.write(b)
            
        with open(self.ppc_nm, 'w') as outfile:
            outfile.write(c)    
            
        os.system('jupyter nbconvert --config='+self.conf_nm + ' --template='+self.tpl_nm)    