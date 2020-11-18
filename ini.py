""" Reads .ini files for the MOTS dataset """

from pathlib import Path

def read_ini(dataset, seq):
    sets = ['train', 'test']
    for this_set in sets:
        ini_file = Path(dataset) / this_set / seq / 'seqinfo.ini'
        if ini_file.is_file():
            text = ini_file.read_text()
            lines = text.split('\n')
            lines = [line for line in lines if line]
            
            data = dict()
            for line in lines:
                splot = line.split('=')
                if len(splot) < 2:
                    continue
                
                key = splot[0]
                val = splot[1]
                
                try:
                    val = int(val)
                except:
                    pass
                
                data[key] = val
            return data
    raise ValueError(f"Could not find any seqinfo.ini files for dataset {dataset} for sequence {seq}")


