import xml.etree.ElementTree as ET
import pandas as pd

# Function to read a local XML file and return a DataFrame
def read_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    print("Root tag:", root.tag)
    print("Attributes:", root.attrib)
    
    all_records = []
    for child in root:
        # print("Child tag:", child.tag, "Attributes:", child.attrib)
        label = child.attrib["index"]
        name = child.attrib["name"]
        record = {"label": label, "name": name}
    
        all_records.append(record)

    df = pd.DataFrame(all_records)
    return df

# Example usage
file_path = "JHU-labels.xml"
df = read_xml(file_path)

# save to CSV
df.to_csv("JHU-ICBM-tract-label_desc.tsv", index=False, sep="\t")