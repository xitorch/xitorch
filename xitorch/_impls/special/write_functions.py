import os
from jinja2 import Environment, FileSystemLoader
import yaml

def main():
    filedir = os.path.dirname(os.path.realpath(__file__))
    absdir = lambda p: os.path.join(filedir, p)

    fgendir = absdir("generated") # directory for generated files
    ftemplatedir = absdir("templates") # directory for the templates
    # functions_file = absdir("functions.json")
    functions_file = absdir("functions.yaml")
    templ_suffix = ".template"

    env = Environment(loader=FileSystemLoader(ftemplatedir))
    if not os.path.exists(fgendir):
        os.mkdir(fgendir)

    with open(functions_file, "r") as f:
        # functions = json.load(f)
        functions = yaml.load(f)

    ftemplates = os.listdir(ftemplatedir)
    for ftempl in ftemplates:
        ftemplate = os.path.join(ftemplatedir, ftempl)
        if not os.path.isfile(ftemplate) or templ_suffix not in ftempl:
            continue

        print("Reading: %s" % ftemplate)
        tmpl = env.get_template(ftempl)
        content = tmpl.render(functions=functions)
        foutname = os.path.split(ftemplate)[-1].replace(templ_suffix, "")
        fout = os.path.join(fgendir, foutname)
        with open(fout, "w") as f:
            f.write(content)

if __name__ == "__main__":
    main()
