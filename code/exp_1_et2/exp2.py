from lxml import etree
import os



xmlFolderPath="../../examples-ls-fr/xml"

citations=[]
for filename in os.listdir(xmlFolderPath):
    file_path = os.path.join(xmlFolderPath, filename)
    if os.path.isfile(file_path) and filename.endswith('.xml'):
        tree = etree.parse(file_path)
        root = tree.getroot()
        for elmt in tree.iter():
            if elmt.tag.endswith("quote"):
                citation={"text":"","lexies":[]}
                if elmt.text!=None:
                    citation["text"]=elmt.text
                for child in elmt:
                    if child.tag.endswith("seg"):
                        citation["lexies"].append(child.attrib["source"].split('/')[-1])
                        if child.text!=None:    
                            citation["text"]+=child.text
                        if child.tail!=None:    
                            citation["text"]+=child.tail
                citations.append(citation)

print(len(citations))
for citation in citations:
    print(citation["text"])
    print(citation["lexies"])
    print("----")



