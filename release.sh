export PYTHON_CMD=python
export DIST_DIR=dist
rm ${DIST_DIR}/*
${PYTHON_CMD} setup.py sdist bdist_wheel
${PYTHON_CMD} -m twine upload ${DIST_DIR}/*
