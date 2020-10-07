import os
from jinja2 import Template
import yaml

def main():
    filedir = os.path.dirname(os.path.realpath(__file__))
    absdir = lambda p: os.path.join(filedir, p)

    fgendir = absdir("generated") # directory for generated files
    ftemplatedir = absdir("templates") # directory for the templates
    functions_yaml = absdir("functions.yml")
    templ_suffix = ".templ"

    if not os.path.exists(fgendir):
        os.mkdir(fgendir)

    with open(functions_yaml, "r") as f:
        functions = yaml.load(f, Loader=yaml.Loader)

    ftemplates = os.listdir(ftemplatedir)
    for ftempl in ftemplates:
        ftemplate = os.path.join(ftemplatedir, ftempl)
        if not os.path.isfile(ftemplate) or templ_suffix not in ftempl:
            continue

        print("Reading: %s" % ftemplate)
        with open(ftemplate, "r") as f:
            tmpl = Template(f.read())

        content = tmpl.render(functions=functions)
        foutname = os.path.split(ftemplate)[-1].replace(templ_suffix, "")
        fout = os.path.join(fgendir, foutname)
        with open(fout, "w") as f:
            f.write(content)

if __name__ == "__main__":
    main()
