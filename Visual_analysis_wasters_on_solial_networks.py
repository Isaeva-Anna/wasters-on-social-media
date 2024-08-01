{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.13","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"none","dataSources":[{"sourceId":9073785,"sourceType":"datasetVersion","datasetId":5473483}],"dockerImageVersionId":30746,"isInternetEnabled":false,"language":"python","sourceType":"notebook","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"from sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.model_selection import GridSearchCV, cross_val_score\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.tree import DecisionTreeClassifier\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nfrom matplotlib import pyplot as plt\nimport seaborn as sns\nimport os\n\n","metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","execution":{"iopub.status.busy":"2024-08-01T13:00:33.215268Z","iopub.execute_input":"2024-08-01T13:00:33.215794Z","iopub.status.idle":"2024-08-01T13:00:33.225626Z","shell.execute_reply.started":"2024-08-01T13:00:33.215758Z","shell.execute_reply":"2024-08-01T13:00:33.223871Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"df = pd.read_csv('/kaggle/input/time-wasters-on-social-media/Time-Wasters on Social Media.csv')\ndf.head()","metadata":{"execution":{"iopub.status.busy":"2024-08-01T13:00:35.017532Z","iopub.execute_input":"2024-08-01T13:00:35.017987Z","iopub.status.idle":"2024-08-01T13:00:35.060259Z","shell.execute_reply.started":"2024-08-01T13:00:35.017957Z","shell.execute_reply":"2024-08-01T13:00:35.058822Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"df.dtypes","metadata":{"execution":{"iopub.status.busy":"2024-08-01T12:41:27.037416Z","iopub.execute_input":"2024-08-01T12:41:27.037836Z","iopub.status.idle":"2024-08-01T12:41:27.050119Z","shell.execute_reply.started":"2024-08-01T12:41:27.037803Z","shell.execute_reply":"2024-08-01T12:41:27.048505Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"df['Income'].plot(kind = 'density')","metadata":{"execution":{"iopub.status.busy":"2024-08-01T12:23:04.282675Z","iopub.execute_input":"2024-08-01T12:23:04.283177Z","iopub.status.idle":"2024-08-01T12:23:04.687591Z","shell.execute_reply.started":"2024-08-01T12:23:04.283141Z","shell.execute_reply":"2024-08-01T12:23:04.686195Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"_, axes = plt.subplots(2, 2, sharey=True, figsize=(10, 4))\n\n\nsns.countplot(x=\"Sex\", hue=\"Survived\", data=data, ax=axes[0, 0])\nsns.countplot(x=\"Parch\", hue=\"Survived\", data=data, ax=axes[1, 0]) # количество детей \nsns.countplot(x=\"Embarked\", hue=\"Survived\", data=data, ax=axes[0, 1]) \nsns.countplot(x=\"SibSp\", hue=\"Survived\", data=data, ax=axes[1, 1]) # количество братьев и сестер  ","metadata":{},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"countries = df['Location'].unique()\n\ncountries_list = countries.tolist()\nprint(\"Уникальные страны в колонке 'Location':\")\nprint(countries_list)","metadata":{"execution":{"iopub.status.busy":"2024-08-01T13:00:44.695627Z","iopub.execute_input":"2024-08-01T13:00:44.696106Z","iopub.status.idle":"2024-08-01T13:00:44.705709Z","shell.execute_reply.started":"2024-08-01T13:00:44.696070Z","shell.execute_reply":"2024-08-01T13:00:44.703510Z"},"trusted":true},"execution_count":null,"outputs":[]},{"cell_type":"code","source":"plt.figure(figsize = (16,8))\nx = []\nfor i in countries_list:\n    x.append(list(df[df['Location'] == i]['Income']))\n\n# Make the histogram using a list of lists\n# Normalize the flights and assign colors and names'''\nplt.hist(x, bins = int(5), label=countries_list)\n\n# Plot formatting\nplt.legend()\nplt.xlabel('Income')\nplt.ylabel('number of people')\n","metadata":{"execution":{"iopub.status.busy":"2024-08-01T13:05:37.875904Z","iopub.execute_input":"2024-08-01T13:05:37.877431Z","iopub.status.idle":"2024-08-01T13:05:38.584269Z","shell.execute_reply.started":"2024-08-01T13:05:37.877383Z","shell.execute_reply":"2024-08-01T13:05:38.582494Z"},"trusted":true},"execution_count":null,"outputs":[]}]}