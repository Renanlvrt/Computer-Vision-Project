#Use command line
echo "Make sure that you used the command line not powershell nor WSL"
myenv\Scripts\activate
echo "activated the virtual environement!"

echo "==============python version used================"
python --version


python -m venv myenv
myenv\Scripts\activate  # on Windows
pip install -r requirements.txt