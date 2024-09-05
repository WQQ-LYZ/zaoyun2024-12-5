import streamlit as st
import pickle
import pandas as pd
from sklearn import preprocessing
import numpy as np
import xgboost as xgb
from pycaret.classification import *
st.set_page_config(
    page_title="HFpEF PhenoRiskAssist——Precision Medicine Starts Here",  # 页面标题
    page_icon="heart_icon.png",  # 页面图标
    layout='wide',
)
st.markdown("""
    <style>
    .css-18e3th9 {
        padding: 0; /* 取消页面内的默认填充 */
    }
    .css-1d391kg {
        padding: 0; /* 取消列之间的默认填充 */
    }

    </style>
    """, unsafe_allow_html=True)
col_form, col_form1 = st.columns([3, 2])
predict_result_code_FULL = 0
with (col_form):


    # 运用表单和表单提交按钮
    with st.form('user_inputs'):

        shifoushengyuguonanhai = st.selectbox('是否生育过男孩', options=['Yes', 'No'])
        shoujiaoyuchengdu = st.selectbox('受教育程度', options=['文盲', '小学','初中','高中或中专','大学及以上'])
        pougongchanhouyindaofenmianshi= st.selectbox('剖宫产后阴道分娩史', options=['Yes', 'No'])
        tangniaobingshi = st.selectbox('糖尿病史', options=['Yes', 'No'])
        bencirenshenqitangniaobing = st.selectbox('本次妊娠期糖尿病', options=['Yes', 'No'])
        gaoxueya = st.selectbox('高血压', options=['慢高', '妊娠期血压','无'])
        qiancipaogongchanzhizhengshifoucunzai = st.selectbox('前次剖宫产指证是否存在', options=['Yes', 'No'])
        yunqianBMI = st.number_input('孕前BMI')
        yunci = st.number_input('孕次')
        jiwangyindaofenmiancishu = st.number_input('既往阴道分娩次数')
        submitted = st.form_submit_button('predict')
        st.markdown("""
                                            <style>
                                            .stButton button {
                                                width: 100%;
                                                font-size: 20px;
                                                margin-top: 25px; /* 调整按钮上方的间距 */
                                            }
                                            </style>
                                            """, unsafe_allow_html=True)

    # 初始化数据预处理格式中岛屿相关的变量
if shifoushengyuguonanhai == 'Yes':
    shifoushengyuguonanhai = 1
elif shifoushengyuguonanhai == 'No':
    shifoushengyuguonanhai = 0

if shoujiaoyuchengdu == '文盲':
    shoujiaoyuchengdu = 1
elif shoujiaoyuchengdu == '小学':
    shoujiaoyuchengdu = 2
elif shoujiaoyuchengdu == '初中':
    shoujiaoyuchengdu = 3
elif shoujiaoyuchengdu == '高中或中专':
    shoujiaoyuchengdu = 4
elif shoujiaoyuchengdu == '大学及以上':
    shoujiaoyuchengdu = 5

if pougongchanhouyindaofenmianshi == 'Yes':
    pougongchanhouyindaofenmianshi = 1
elif pougongchanhouyindaofenmianshi == 'No':
    pougongchanhouyindaofenmianshi = 0

if tangniaobingshi == 'Yes':
    tangniaobingshi = 1
elif tangniaobingshi == 'No':
    tangniaobingshi = 0

if bencirenshenqitangniaobing == 'Yes':
    bencirenshenqitangniaobing = 1
elif bencirenshenqitangniaobing == 'No':
    bencirenshenqitangniaobing = 0

if gaoxueya == '慢高':
    gaoxueya = 1
elif gaoxueya == '妊娠期血压':
    gaoxueya = 2
elif gaoxueya == '无':
    gaoxueya = 3

if qiancipaogongchanzhizhengshifoucunzai == 'Yes':
    qiancipaogongchanzhizhengshifoucunzai = 1
elif qiancipaogongchanzhizhengshifoucunzai == 'No':
    qiancipaogongchanzhizhengshifoucunzai = 0



category_data = [shifoushengyuguonanhai, shoujiaoyuchengdu, pougongchanhouyindaofenmianshi,
                 tangniaobingshi, bencirenshenqitangniaobing, gaoxueya, qiancipaogongchanzhizhengshifoucunzai]

numerical_data = [yunqianBMI, yunci, jiwangyindaofenmiancishu]

numerical_data = np.array(numerical_data).reshape(1, -1)  # 转换为二维数组

min_max_scaler = preprocessing.MinMaxScaler()
numerical_data = min_max_scaler.fit_transform(numerical_data)

format_data =     category_data + numerical_data.flatten().tolist()
format_data=np.array(format_data)
columns = [
    'shifoushengyuguonanhai', 'shoujiaoyuchengdu', 'yunci',
    'pougongchanhouyindaofenmianshi', 'tangniaobingshi',
    'bencirenshenqitangniaobing', 'gaoxueya',
    'qiancipaogongchanzhizhengshifoucunzai', 'yunqianBMI',
    'jiwangyindaofenmiancishu'
]
format_data = pd.DataFrame([format_data], columns=columns)
# format_data = pd.DataFrame(format_data, index=[0])  # 假设我们只有一行数据，所以index设置为[0]
print(format_data)
#for item in format_data:
#    print(type(item))
format_data.iloc[:, :7] = format_data.iloc[:, :7].astype(int)
format_data.iloc[:, 7:] = format_data.iloc[:, 7:].astype(float)
# print("**********************")
# print(format_data)
#print(format_data.dtypes)
with (col_form1):
    # 使用pickle的load方法从磁盘文件反序列化加载一个之前保存的随机森林模型对象

    with open('xgboost_model.pkl', 'rb') as f:
        rfc_model = pickle.load(f)
    # 使用pickle的load方法从磁盘文件反序列化加载一个之前保存的映射对象
    # with open('output_uniques.pkl', 'rb') as f:
    #     output_uniques_map = pickle.load(f)

    if submitted:

        # 使用模型对格式化后的数据format_data进行预测,返回预测的类别代码
        predict_result_code = rfc_model.predict(format_data)

        # 根据预测结果输出对应的企鹅物种名称
        if submitted:

            # 使用模型对格式化后的数据format_data进行预测,返回预测的类别代码
            predict_result_code = rfc_model.predict(format_data)

            # 根据预测结果输出对应的企鹅物种名称
            if predict_result_code == 1:
                predict_result_code_FULL = 1
                st.write(f'建议分娩方式为：剖宫产')
            elif predict_result_code == 0:
                predict_result_code_FULL = 0
                st.write(f'建议分娩方式为：顺产')
