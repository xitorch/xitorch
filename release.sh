# steps:
# * check the version first
# * create a token in test.pypi.org or pypi.org
# * execute these commands below

export PYTHON_CMD=python
export PIP=pip
export DIST_DIR=dist
rm -f ${DIST_DIR}/*
${PYTHON_CMD} -m ${PIP} install --upgrade pip
${PYTHON_CMD} -m ${PIP} install --upgrade build
${PYTHON_CMD} -m ${PIP} install --upgrade twine
${PYTHON_CMD} -m build
# for test
${PYTHON_CMD} -m twine upload --repository testpypi ${DIST_DIR}/*
# for release
${PYTHON_CMD} -m twine upload ${DIST_DIR}/*
