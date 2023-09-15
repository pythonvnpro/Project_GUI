import os
from datetime import datetime
from datetime import timedelta
import streamlit as st
from streamlit.components.v1 import iframe
# from streamlit import caching
import base64 as b64
from PIL import Image
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import re
from utils import _initialize_spark
import sys
from pyspark import SparkContext
### Functions: Chỉ cho hiện những hình nằm trong phạm vi cấu hình fr - to
def project2_show_range_img(directory, fr=1, to=24):
    # Use os.listdir to get all files in the directory
    files = os.listdir(directory)

    def sort_key(file_name):
        # Extract the number from the file name (much number)
        number = int(file_name.split('.')[0])
        return number

    # Sort the list of files using the custom sorting function
    files = sorted(files, key=sort_key)
    print(files)
    # Filter the list to only include .PNG files with file names between fr and to
    images = [file for file in files if file.endswith('PNG') and fr <= int(file.split('.')[0]) <= to]
    return images
### Functions: Cho tùy chỉnh tùy biến danh sách các hình muốn hiển thị không cần theo quy luật nào.
def project2_show_list_img(directory, list=[10, 5, 8]):
    # Use os.listdir to get all files in the directory
    files = os.listdir(directory)

    def sort_key(file_name):
        # Extract the number from the file name
        number = int(file_name.split('.')[0])
        return number

    # Sort the list of files using the custom sorting function
    files = sorted(files, key=sort_key)
    print(files)
    # Filter the list to only include .PNG files with file names in list
    images = [file for file in files if file.endswith('PNG') and int(file.split('.')[0]) in list]
    return images

### markdown: right
from pathlib import Path
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = b64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path, width, height):
    img_html = "<img src='data:image/png;base64, {}' class='img-fluid' width='{}' height='{}'>".format(
        img_to_bytes(img_path), width, height
    )
    return img_html

st.markdown("<p style='text-align: center; color: grey;'>"+img_to_html('images/CapstoreProject.png', 700, 350)+"</p>", unsafe_allow_html=True)
#----------------------------------------------------------------------------------------------------------------------------------#

# Tạo cột bên trái cho menu
left_column = st.sidebar
# Chèn đoạn mã HTML để tùy chỉnh giao diện của chữ "Chọn dự án" : left_column.markdown('<span style="font-weight: bold; color: blue;">Chọn dự án</span>', unsafe_allow_html=True)
# Tạo danh sách các dự án
projects =  ['Project 1: Customer Segmentation','Project 2: Recommendation System', 'Project 3: Sentiment Analysis']

# Tạo menu dropdown list cho người dùng lựa chọn dự án
project = left_column.selectbox(":blue[**Select project:**]", projects, index=1)

# Lưu trữ chỉ số index của dự án được chọn
project_num = projects.index(project) + 1


def highlight_rows_even_odd(row):
    if row.name % 2 == 0:
        return ['background-color: lightcoral']*len(row)
    else:
        return ['background-color: lightgray']*len(row)

# Chọn dự án
if project_num == 1:
    # Hiển thị tên của dự án 
    st.subheader("Project 1: Customer Segmentation")
    # Hiển thị danh sách các tùy chọn cho người dùng lựa chọn từng bước trong dự án
    step = left_column.radio('', ['Business Understanding', 'Preprocessing + EDA', 'Applicable models', 'Prediction'])
    
    # Xử lý sự kiện khi người dùng lựa chọn từng mục trong danh sách và hiển thị hình ảnh tương ứng
    if step == 'Business Understanding':
        st.image(f'Project_{project_num}/images/a{project_num}.jpg')
    elif step == 'Preprocessing + EDA':
        st.image(f'Project_{project_num}/images/b{project_num}.jpg')
    elif step == 'Applicable models':
        st.image(f'Project_{project_num}/images/c{project_num}.jpg')
    elif step == 'Prediction':
        st.image(f'Project_{project_num}/images/d{project_num}.jpg')
elif project_num == 2:
    # Hiển thị tên của dự án 
    st.subheader("Project 2: Recommendation System")
    # Hiển thị thời gian bắt đầu Streamlit bắt đầu dự án, để kiểm tra chéo với thời gian kết thúc.
    first_step = datetime.now()
    st.write(f"Now is: {first_step}")

### START: TRÁNH LÀM CHẬM HỆ THỐNG DO PHẢI XỬ LÝ LẠI MỖI KHI CHỌN CÁC CHỨC NĂNG ###
    # 1. Read data
    if 'product_raw' not in st.session_state:
        # Chưa có, đọc dữ liệu và lưu trữ lại trong phiên làm việc
        product_raw = pd.read_csv(f'Project_{project_num}/Data/ProductRaw.csv', encoding='utf-8')
        st.session_state['product_raw'] = product_raw
    else:
        # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
        print("Dữ liệu dataframe product_raw đã có sắn, chỉ việc lấy ra sử dụng")
        product_raw = st.session_state['product_raw']
    if 'review_raw' not in st.session_state:
        # Chưa có, đọc dữ liệu và lưu trữ lại trong phiên làm việc
        review_raw = pd.read_csv(f'Project_{project_num}/Data/ReviewRaw.csv', encoding='utf-8')
        st.session_state['review_raw'] = review_raw
    else:
        # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
        print("Dữ liệu dataframe review_raw đã có sắn, chỉ việc lấy ra sử dụng")
        review_raw = st.session_state['review_raw']

    # Kiểm tra dataframe data 
    if 'data' not in st.session_state:
        # Chưa có, đọc dữ liệu và lưu trữ lại trong phiên làm việc
        # data = pd.read_csv(f'Project_{project_num}/Data/3_Review_Product.csv', encoding='utf-8') # Do file lớn quá hosting không cho upload file này
        review_clean = pd.read_csv(f'Project_{project_num}/Data/3_Review.csv', encoding='utf-8')
        product_clean = pd.read_csv(f'Project_{project_num}/Data/3_Product.csv', encoding='utf-8')
        data = review_clean.merge(product_clean, how= "left", left_on= "product_id", right_on= "item_id")
        st.session_state['data'] = data
        st.session_state['product_clean'] = product_clean
    else:
        # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
        print("Dữ liệu dataframe data đã có sắn, chỉ việc lấy ra sử dụng")
        data = st.session_state['data']
        product_clean= st.session_state['product_clean']

    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding='utf-8')
        data.to_csv(f'Project_{project_num}/Data/3_Review_Product_new.csv', index=False)
        # Cập nhật lại data sẽ được lưu trữ ở phiên làm việc
        st.session_state['data'] = data

### END: TRÁNH LÀM CHẬM HỆ THỐNG DO PHẢI XỬ LÝ LẠI MỖI KHI CHỌN CÁC CHỨC NĂNG ###

    # Hiển thị danh sách các tùy chọn cho người dùng lựa chọn từng bước trong dự án
    step = left_column.radio('', ['Business Understanding', 'Preprocessing + EDA', 'Applicable models', 'Recommendation'])
    # Xử lý sự kiện khi người dùng lựa chọn từng mục trong danh sách và hiển thị hình ảnh tương ứng
    if step == 'Business Understanding':
        st.image(f'Project_{project_num}/images/a{project_num}.jpg')
        # Đường dẫn đến files png
        directory = f'Project_{project_num}/images/slides'
        images = project2_show_range_img(directory, 1, 4)
        # Loop through the images and display them using st.image
        for image in images:
            img = Image.open(os.path.join(directory, image))
            print(img)
            st.image(img)
    elif step == 'Preprocessing + EDA':
        # Đường dẫn đến files png
        directory = f'Project_{project_num}/images/slides'
        images = project2_show_range_img(directory, 5, 5)
                # Loop through the images and display them using st.image
        for image in images:
            img = Image.open(os.path.join(directory, image))
            print(img)
            st.image(img)
        not_existed = review_raw[~review_raw['product_id'].isin(product_raw['item_id'])]
        st.write(f"### Kiểm tra thô ban đầu cho thấy có {len(not_existed)} dữ liệu mồ côi")
        st.dataframe(not_existed.style.apply(highlight_rows_even_odd, axis=1))
        st.write(f"### Số lượng được đánh giá ảo lần lượt là:")
        st.dataframe(not_existed['rating'].value_counts())
        st.write(f"\n### Dùng dữ liệu sau khi xử lý và loại bỏ {len(not_existed)} dữ liệu gây nhiễu")
        st.write(f"\n#### Chi tiết kết quả đánh giá của review (rating)")
        mapping = {'Rất hài lòng':5, 'Hài lòng':4, 'Trung bình':3, 'Không hài lòng':2, 'Tệ':1}
        for k, v in mapping.items():
            st.write(k + ',')
            st.write(data[data['rating'] == v].drop(['customer_id','product_id','item_id','product_rating'], axis=1).describe().T)
            st.write()
        st.markdown("**Nhận xét**")
        st.markdown("Trong các thang điểm được đánh giá trên sản phẩm thì:")
        st.markdown("- price cao nhất 51990000.0 nhóm rating = 5")
        st.markdown("- price thấp nhất 7000.0 rơi vào nhóm rating = [5, 4, 1]")
        st.markdown("- list_price cao nhất 82990000.0 nhóm rating = 5")
        st.markdown("- list_price thấp nhất 12000.0 cũng rơi vào nhóm rating = [5, 4, 1]")
        # Create a figure and axis
        fig, ax = plt.subplots()
        # Iterate over the mapping and plot the data
        for k, v in mapping.items():
            data_rating = data[data['rating'] == v].drop(['customer_id','product_id','item_id','product_rating'], axis=1).describe().T
            bar_data = pd.DataFrame({'category': [k], 'count': [int(data_rating['count'].iloc[0])]}).sort_values(by='count', ascending=False)
            ax.bar(k, bar_data['count'], label=k)
            # Add text annotations for the counts
            ax.text(k, bar_data['count'].iloc[0], bar_data['count'].iloc[0], ha='center', va='bottom')
        
        # Add labels and title
        ax.set_xlabel('Rating')
        ax.set_ylabel('Number')
        ax.set_title('Total Value by Rating')
        # Display the legend
        ax.legend()
        # Display the chart in Streamlit
        st.pyplot(fig)

        st.write(f"\n#### Chi tiết kết quả đánh giá cho product (product_rating)")
        mapping = {'Rất hài lòng':5, 'Hài lòng':4, 'Trung bình':3, 'Không HL':2, 'Tệ':1, 'Chưa đ.giá':0}
        for k, v in mapping.items():
            st.write(k + ',')
            st.write(data[data['product_rating'] == v].drop(['customer_id','product_id','item_id','rating'], axis=1).describe().T)
            st.write()

        st.markdown("**Nhận xét**")
        st.markdown("Trong các thang điểm được đánh giá trên sản phẩm thì:")
        st.markdown("- price cao nhất 62690000.0 nhóm product_rating = 0")
        st.markdown("- price thấp nhất 9200.0 rơi vào nhóm product_rating = 4")
        
        st.markdown("- list_price cao nhất 82990000.0 nhóm product_rating = 5")
        st.markdown("- list_price thấp nhất 14000.0 cũng rơi vào nhóm product_rating = 5")
        # Create a figure and axis
        fig, ax = plt.subplots()
        # Iterate over the mapping and plot the data
        for k, v in mapping.items():
            product_rating = data[data['product_rating'] == v].drop(['customer_id','product_id','item_id','rating'], axis=1).describe().T
            bar_data = pd.DataFrame({'category': [k], 'count': [int(product_rating['count'].iloc[0])]}).sort_values(by='count', ascending=False)
            ax.bar(k, bar_data['count'], label=k)
            # Add text annotations for the counts
            ax.text(k, bar_data['count'].iloc[0], bar_data['count'].iloc[0], ha='center', va='bottom')

        # Add labels and title
        ax.set_xlabel('Rating')
        ax.set_ylabel('Number')
        ax.set_title('Total Value by Rating')
        # Display the legend
        ax.legend(loc='upper center')
        # Display the chart in Streamlit
        st.pyplot(fig)

    #### 10 Khách hàng đánh giá nhiều nhất, ít nhất
        st.markdown("#### 10 Khách hàng đánh giá nhiều nhất, ít nhất")
        customers_by_rating = data.groupby('customer_id')['rating'].value_counts().reset_index(name='Count')
        plot_top_10 = customers_by_rating.sort_values(by=['Count','customer_id'], ascending= [False,False]).head(10)
        plot_down_10 = customers_by_rating.sort_values(by=['Count','customer_id',], ascending= [True,True]).head(10)
        #Vẽ biểu đồ
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
        # Biểu đồ: Top 10 sản phẩm được khách hàng mua nhiều nhất
        splot1 = sns.barplot(y= plot_top_10["Count"], x= plot_top_10["customer_id"], hue='rating', data= plot_top_10 , color = 'blue', palette = 'hls', ax=axs[0], dodge=True, order = plot_top_10["customer_id"].unique())
        for g in splot1.patches:
            splot1.annotate(format(g.get_height(), '.0f'),
                        (g.get_x() + g.get_width() / 2., g.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 10),
                        textcoords = 'offset points')
        axs[0].set_ylim(0, 60)
        axs[0].set_ylabel('Số lần đánh giá')
        axs[0].set_xlabel('Mã khách hàng')

        axs[0].set_title('Top 10 khách hàng đánh giá nhiều nhất', fontsize=16)

        # Biểu đồ: Top 10 sản phẩm được khách hàng mua ít nhất
        splot2 = sns.barplot(y= plot_down_10["Count"], x= plot_down_10["customer_id"], hue='rating', data= plot_down_10 , color = 'blue', palette = 'hls', ax=axs[1], dodge=True, order = plot_down_10["customer_id"].unique())
        for g in splot2.patches:
            splot2.annotate(format(g.get_height(), '.0f'),
                        (g.get_x() + g.get_width() / 2., g.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 10),
                        textcoords = 'offset points')
        axs[1].set_ylim(0, 10)
        axs[1].set_ylabel('Số lần đánh giá')
        axs[1].set_xlabel('Mã khách hàng')
        axs[1].set_title('Top 10 khách hàng đánh giá ít nhất', fontsize=16)
        plt.tight_layout()
        st.pyplot(fig)
    #### 10 Sản phẩm được mua nhiều, ít nhất
        st.markdown("#### 10 Sản phẩm được mua nhiều, ít nhất")
        plot= data["product_name"].value_counts().reset_index()
        plot.columns = ["product_name", "Count"]
        plot_top_10 = plot.sort_values(by=['Count'], ascending= [False]).head(10)
        plot_down_10 = plot.sort_values(by=['Count'], ascending= [True]).head(10)

        #Vẽ biểu đồ
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,6))

        # Biểu đồ: Top 10 sản phẩm được khách hàng mua nhiều nhất
        splot1 = sns.barplot(y= plot_top_10["Count"], x= plot_top_10["product_name"], data= plot_top_10 , color = 'blue', palette = 'hls', ax=axs[0])
        for g in splot1.patches:
            splot1.annotate(format(g.get_height(), '.0f'),
                        (g.get_x() + g.get_width() / 2., g.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 10),
                        textcoords = 'offset points')
        axs[0].set_ylim(0, 5000)
        axs[0].set_ylabel('Số lượng')
        axs[0].set_xlabel('')
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation='vertical')
        axs[0].set_title('Top 10 sản phẩm được khách hàng mua nhiều nhất', fontsize=16)

        # Biểu đồ: Top 10 sản phẩm được khách hàng mua ít nhất
        splot2 = sns.barplot(y= plot_down_10["Count"], x= plot_down_10["product_name"], data= plot_down_10 , color = 'blue', palette = 'hls', ax=axs[1])
        for g in splot2.patches:
            splot2.annotate(format(g.get_height(), '.0f'),
                        (g.get_x() + g.get_width() / 2., g.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 10),
                        textcoords = 'offset points')
        axs[1].set_ylim(0, 2)
        axs[1].set_ylabel('Số lượng')
        axs[1].set_xlabel('')
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation='vertical')
        axs[1].set_title('Top 10 sản phẩm được khách hàng mua ít nhất', fontsize=16)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("**Nhận xét**")
        st.markdown("- Các mặt hàng chưa được đánh giá là : Remote dùng cho điều hòa Panasonic 1 chiều, [Miếng dán màn hình] Kính cường lực chống nhìn trộm dành cho Iphone 6/7/8/X/11 6Plus 7Plus 8Plus XS MAX Iphone 11 Pro Max chất lượng, .....")
        st.markdown("- Các mặt hàng được đánh giá đông nhất là: Chuột Không Dây Logitech M331 Silent - Hàng Chính Hãng, Pin Sạc Dự Phòng Xiaomi Redmi 20000mAh PB200LMZ Tích Hợp Cổng USB Type - C In Hỗ Trợ Sạc Nhanh 18W - Hàng Chính Hãng......")
    elif step == 'Applicable models':
        # Đường dẫn đến files png
        directory = f'Project_{project_num}/images/slides'
        images = project2_show_list_img(directory, [6,7,14,15,16,17,18,19,20,22])
        # Loop through the images and display them using st.image
        for image in images:
            img = Image.open(os.path.join(directory, image))
            print(img)
            st.image(img)
    elif step == 'Recommendation':
        t0_recommendation = datetime.now()
        tracking_num= 2
        t0_config = datetime.now()
        # Create a expander
        with st.expander('**Configuration**'):
            # Create an input box with a default value of 5
            num_rows = st.number_input('**Setup the number of recommended results**', min_value=1, max_value= 50, value=2)
            st.write(f"Run time is {str(datetime.now() - t0_config)}")

        # st.write(f"num_rows: **{num_rows}**") # Debug
        # Load the customer data
        @st.cache_data
        def get_unique_customer_ids(data):
            return data['customer_id'].unique()
        t0_welcome = datetime.now()
        # Create a selectbox with cached data
        customer_id_login = st.selectbox('**Customer id to log in to the system**', get_unique_customer_ids(data))
        # customer_id = st.selectbox('**Select Customer ID**', data['customer_id'].unique())
        st.write(f"Preparation time to welcome the return of customer number **{customer_id_login}** is **{str(datetime.now() - t0_welcome)}**")
        def text_field(label, columns=None, **input_params):
            c1, c2 = st.columns(columns or [1, 4])

            # Hiển thị tên trường với một số căn chỉnh
            c1.markdown("##")
            c1.markdown(label)

            # Đặt một tham số khóa mặc định để tránh lỗi khóa trùng lặp
            input_params.setdefault("key", label)

            # Chuyển tiếp các tham số đầu vào văn bản
            return c2.text_input(label="", **input_params,label_visibility="hidden")
        ## pyspark ALS

        on_Collaborative_filtering = st.toggle(':orange[**Activate feature \"Collaborative filtering\"**]',value= False)
        on_Content_based_filtering = st.toggle(':blue[**Activate feature \"Content-based filtering\"**]',value= False)
        if on_Collaborative_filtering:  # Cho phép chạy
            # DO THỜI ĐIỂM LÀM VIỆC VỚI STREAMLIT SERVER DOWN LIÊN TỤC NÊN PHẢI VIẾT THÊM HÀM NÀY RETRY TRONG _initialize_spark
            if 'spark' not in globals():
                result = _initialize_spark()
                if result is not None:
                    spark, sc = result
                else:
                    # Handle the case where _initialize_spark() returns None
                    # For example, you could print an error message and exit the program
                    print("Error: _initialize_spark() returned None")
                    sys.exit(1)
            else:
                spark = globals()['spark']
                sc = spark.sparkContext
                # Kiểm tra xem liệu SparkContext đã tồn tại hay chưa
                if 'sc' not in globals():
                    sc = SparkContext.getOrCreate()
                else:
                    sc = globals()['sc']                    
            from pyspark.sql.functions import *
            from pyspark.sql.types import *
            from pyspark.ml.feature import Binarizer, Bucketizer, OneHotEncoder, StringIndexer, VectorAssembler
            from pyspark.ml.linalg import Vectors
            # Creating ALS model and fitting data
            from pyspark.ml.evaluation import RegressionEvaluator
            from pyspark.ml.recommendation import ALS
            from pyspark.ml.recommendation import ALSModel
            model_t = ALSModel.load(f'Project_{project_num}/als_models/als')
            if 'data_indexed' not in st.session_state:
                data_sub = spark.createDataFrame(data[['product_id','product_name', 'rating', 'customer_id']])
                # Đảm cột rating nội dung là kiểu double 
                data_sub = data_sub.withColumn("rating", data_sub["rating"].cast(DoubleType()))
                users = data_sub.select("customer_id").distinct().count()
                products = data_sub.select("product_id").distinct().count()
                numerator = data_sub.count()
                # Number of ratings matrix could contain if no empty cells
                denominator = users * products
                #Calculating sparsity
                sparsity = 1 - (numerator*1.0 / denominator)
                # Converting String to index
                from pyspark.ml import Pipeline
                # Create an indexer
                indexer = StringIndexer(inputCol='product_id', 
                                        outputCol='product_id_idx')

                # Indexer identifies categories in the data
                indexer_model = indexer.fit(data_sub)

                # Indexer creates a new column with numeric index values
                data_indexed = indexer_model.transform(data_sub)

                # Repeat the process for the other categorical feature
                indexer1 = StringIndexer(inputCol='customer_id', 
                                        outputCol='customer_id_idx')
                indexer1_model = indexer1.fit(data_indexed)
                data_indexed = indexer1_model.transform(data_indexed)
                # Gán vào session_state để khi dùng kiểm tra có không phải đi làm lại các bước trên.
                st.session_state['data_indexed'] = data_indexed
            else:
                # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
                print("Dữ liệu dataframe data_indexed đã có sắn, chỉ việc lấy ra sử dụng")
                data_indexed = st.session_state['data_indexed']
            # Nắn dữ liệu cho báo cáo
            df_customer_customer_idx = data_indexed.select('customer_id_idx', 'customer_id').distinct()
            df_product_id_product_id_idx = data_indexed.select('product_id_idx', 'product_id','product_name').distinct()

            # st.write(f"tracking_num: **{tracking_num}**")  # Debug
            if 'user_recs' not in st.session_state or tracking_num != num_rows:
                # Gợi ý theo customer_id
                user_recs = model_t.recommendForAllUsers(num_rows)
                # Nối lại mã khách hàng và hiển thị
                new_user_recs = user_recs.join(df_customer_customer_idx, on=['customer_id_idx'], how='left')
                st.session_state['user_recs'] = user_recs
                # Cập nhật lại để đảm bảo chỉ khi có thay đổi số dòng kết quả gợi ý mới vào đây để load và lại cập nhật giá trị.
                tracking_num = num_rows
            else:
                # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
                print("Dữ liệu dataframe user_recs đã có sắn, chỉ việc lấy ra sử dụng")
                user_recs = st.session_state['user_recs']
                # Nối lại mã khách hàng và hiển thị
                new_user_recs = user_recs.join(df_customer_customer_idx, on=['customer_id_idx'], how='left')

            ## DO ĐƯA LÊN STREAMLIT CHẠY KHÔNG NỔI NÊN TẠM THỜI ĐÓNG LẠI PHẦN NÀY: GỢI Ý DỰA TRÊN MÃ SẢN PHẨM = ALS
            # if 'product_recs' not in st.session_state or tracking_num != num_rows:
            #     # Gợi ý theo product_id
            #     product_recs = model_t.recommendForAllItems(num_rows)
            #     # Nối lại mã sản phẩm và hiển thị
            #     new_product_recs = product_recs.join(df_product_id_product_id_idx, on=['product_id_idx'], how='left')
            #     st.session_state['product_recs'] = product_recs
            # else:
            #     # Dữ liệu đã có sắn, chỉ việc lấy ra sử dụng
            #     print("Dữ liệu dataframe product_recs đã có sắn, chỉ việc lấy ra sử dụng")
            #     product_recs = st.session_state['product_recs']
            #     # Nối lại mã sản phẩm và hiển thị
            #     new_product_recs = product_recs.join(df_product_id_product_id_idx, on=['product_id_idx'], how='left')

            #DO ĐƯA LÊN STREAMLIT CHẠY: THỬ GIẢI PHÁP CHUYỂN pyspark dataframe -> pandas dataframe PHÁ SẢN HOÀN TOÀN TỐN BỘ NHỚ HƠN
            def als_recommandations_pandas(top_user, show_top, filter_score):
                    # Chuyển đổi new_user_recs sang DataFrame pandas
                    new_user_recs_pd = new_user_recs.toPandas()
                    
                    # Lọc dữ liệu bằng cách sử dụng các phương thức của pandas
                    find_user_rec = new_user_recs_pd[new_user_recs_pd['customer_id'] == top_user]
                    if not find_user_rec.empty:
                        st.markdown("\n+ Recommendations for customers: ", top_user)
                        user = find_user_rec.iloc[0]

                        # Convert list to DataFrame
                        report = pd.DataFrame([user])

                        # Explode the recommendations column into multiple rows
                        report = report.explode('recommendations')

                        # Select the customer_id and create new columns for product_id_idx and rating
                        report['product_id_idx'] = report['recommendations'].apply(lambda x: x.product_id_idx)
                        report['als_score'] = report['recommendations'].apply(lambda x: x.rating)
                        report = report[['customer_id', 'product_id_idx', 'als_score']]

                        ### LỌC ĐỀ XUẤT DỰA TRÊN NGƯỠNG
                        report_filtered = report[report.als_score >= filter_score]

                        # Join df_filtered with df_product_id_product_id_idx on product_id_idx
                        df_product_id_product_id_idx_pd = df_product_id_product_id_idx.toPandas()
                        report_filtered = pd.merge(report_filtered, df_product_id_product_id_idx_pd, on='product_id_idx', how='left')
                        report_filtered = report_filtered[['customer_id', 'product_id', 'product_name', 'als_score']]
                        report_filtered = report_filtered.sort_values(by='als_score', ascending=False)
                        st.dataframe(report_filtered.head(num_rows).style.apply(highlight_rows_even_odd, axis=1))
                        # report_filtered.show(show_top, truncate=False)
                    
                    else:
                        st.markdown("Nothing to recommend to customers:", top_user)

            def als_recommandations(top_user, show_top, filter_score):
                # find_user_rec = new_user_recs.filter(new_user_recs['customer_id'] == top_user)
                # if find_user_rec.count() > 0:
                ## Thay 2 dòng trên = 2 dòng dưới này cải thiện tốc độ giảm hơn 1/2 thời gian chạy
                find_user_rec = new_user_recs.filter(new_user_recs['customer_id'] == lit(top_user))
                find_user_rec = find_user_rec.persist()
                if find_user_rec.limit(1).collect():
                    st.markdown("\n+ Recommendations for customer: ")
                    user = find_user_rec.first() 

                    # Convert list to DataFrame
                    report = spark.createDataFrame([user])

                    # Explode the recommendations column into multiple rows
                    report = report.select(col("customer_id"), explode(col("recommendations")).alias("recommendations"))

                    # Select the customer_id and create new columns for product_id_idx and rating
                    report = report.select(col("customer_id"), 
                                col("recommendations").product_id_idx.alias("product_id_idx"),
                                col("recommendations").rating.alias("als_score"))

                    ### LỌC ĐỀ XUẤT DỰA TRÊN NGƯỠNG
                    report_filtered = report.filter(report.als_score >= filter_score)

                # Join df_filtered with df_product_id_product_id_idx on product_id_idx
                    # report_filtered = report_filtered.join(df_product_id_product_id_idx, on=['product_id_idx'], how='left').select("customer_id", "product_id","product_name", "als_score").sort(desc("als_score"))
                    ## Thay dòng trên =  dòng dưới này cải thiện tốc độ giảm hơn 1/2 thời gian chạy
                    report_filtered = report_filtered.join(broadcast(df_product_id_product_id_idx), on=['product_id_idx'], how='left').select("customer_id", "product_id","product_name", "als_score").sort(desc("als_score"))
                    # Show the filtered report
                    report_filtered= report_filtered.toPandas()
                    st.dataframe(report_filtered.head(num_rows).style.apply(highlight_rows_even_odd, axis=1))
                    # report_filtered.show(show_top, truncate=False)
                    
                else:
                    st.write("Nothing to recommend to customer: {top_user}")

            def als_recommandations_return_result(top_user, show_top, filter_score):
                find_user_rec = new_user_recs.filter(new_user_recs['customer_id'] == top_user)
                if find_user_rec.count() > 0:
                    st.markdown("\n+ Recommendations for customers: ", top_user)
                    user = find_user_rec.first() 

                    # Convert list to DataFrame
                    report = spark.createDataFrame([user])

                    # Explode the recommendations column into multiple rows
                    report = report.select(col("customer_id"), explode(col("recommendations")).alias("recommendations"))

                    # Select the customer_id and create new columns for product_id_idx and rating
                    report = report.select(col("customer_id"), 
                                col("recommendations").product_id_idx.alias("product_id_idx"),
                                col("recommendations").rating.alias("als_score"))

                    ### LỌC ĐỀ XUẤT DỰA TRÊN NGƯỠNG
                    report_filtered = report.filter(report.als_score >= filter_score)

                    # Join df_filtered with df_product_id_product_id_idx on product_id_idx
                    report_filtered = report_filtered.join(df_product_id_product_id_idx, on=['product_id_idx'], how='left').select("customer_id", "product_id","product_name", "als_score").sort(desc("als_score"))

                    # Show the filtered report
                    report_filtered= report_filtered.toPandas()
                    st.dataframe(report_filtered.head(num_rows).style.apply(highlight_rows_even_odd, axis=1))
                    # report_filtered.show(show_top, truncate=False)
                    # Return the DataFrame
                    return report_filtered
                    
                else:
                    st.markdown("+ Nothing to recommend to customers:", top_user)
                    return None
            with st.expander('#### **Collaborative filtering**\n(Các bạn feedback thông thường lấy top 5 là  3 phút trên 1 kết quả tìm kiếm)'):                
                # Kiểm tra xem st.session_state.customer_id có tồn tại hay không
                if 'customer_id' not in st.session_state:
                    st.session_state.customer_id = '709310'

                # Kiểm tra xem customer_id có phải là một chuỗi rỗng hay không và thay thế nó bằng một giá trị mới
                if 'customer_id' not in st.session_state or not st.session_state.customer_id:
                    st.session_state.customer_id = '709310'
                elif st.session_state.customer_id == '':
                    st.session_state.customer_id = str(customer_id_login)
                    st.session_state['key'] = str(customer_id_login)

                # Check if the text field is empty: Trong trường hợp người dùng cố tình xóa gây lỗi cho hệ thống thì giải pháp này sẽ giải quyết đúng logic chương trình

                def callback():
                    st.text(st.session_state.customer_id)

                # Gọi hàm text_field để tạo widget text_input
                customer_id = text_field('**Search customer id**', value=str(st.session_state.customer_id), on_change=callback, key='key', autocomplete=None)
                if customer_id != '':
                    st.session_state.customer_id = customer_id
                    t0_als_recommandations = datetime.now()
                    als_recommandations(int(customer_id), int(num_rows), filter_score=0.0)
                    st.write('Total run time Collaborative filtering - Recommendation: \t\t',datetime.now() - t0_als_recommandations)

        if on_Content_based_filtering:
            # ĐÓNG LẠI ĐỂ KHÔNG PHẢI GÁNH THÊM THỜI GIAN CHẠY LẠI => on_Collaborative_filtering
            import gensim
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            from scipy.sparse import vstack
            def get_product_name(product_id):
                product_name= data.loc[data['item_id'] == product_id, ['product_name','group_wt','description_wt']]
                return product_name
            # Create a expander
            with st.expander('#### **Content-based filtering**'):
                tab1, tab2 = st.tabs(["## **Gợi ý - Theo theo nội dung khách hàng nhập vào tìm kiếm**\n**(Cosin)**", "## **Gợi ý - Theo sản phẩm khách hàng đang xem**\n**(Gensim)**"])
                with tab1:
                    # Create 3 columns
                    col1, col2, col3 = st.columns(3)
                    # Create checkboxes for search options
                    search_by_gensim = col1.checkbox(label='Search by Gensim',value=True)
                    search_by_cosine_similarity = col2.checkbox(label='Search by Cosine Similarity*',value=True)
                    search_by_als = col3.checkbox(label='Search by ALS', disabled=True)
                    search_content = text_field('**Search content**', value='Da ')

                    t0_gensim_cosine_als = datetime.now()
                    # Create a search button
                    if st.button('Search'):
                        # Display the search results based on the selected options
                        if search_by_gensim:
                            ## Gensim
                            #1. Load Gensim model
                            # Tải mô hình và từ điển
                            model_gensim = gensim.models.TfidfModel.load(f'Project_{project_num}/gensim_models/tfidf.model')
                            # Đọc từ điển đã được lưu trữ lên dùng
                            dictionary = gensim.corpora.Dictionary.load(f'Project_{project_num}/gensim_models/dictionary.gensim')
                            # Đọc index đã được lưu trữ lên dùng
                            index = gensim.similarities.Similarity.load(f'Project_{project_num}/gensim_models/index_file')
                            def gensim_recommendation(view_product_name, number_re= 3, threshold= 0):
                                '''
                                view_product_name:   sản phẩm đang cần tìm
                                number_re:              trả về bao nhiêu kết quả tìm được tốt nhất
                                threshold:              lọc lại kết quả có điểm >= similarity_score mới trả về
                                '''
                                if len(view_product_name) > 255:
                                    view_product_name = view_product_name[:255]
                                # Rả ra thông tin
                                view_product = view_product_name.lower().split()
                                # Chuyển đổi các từ tìm kiếm thành Vectơ thưa thớt (Convert search words into Sparse Vectors)
                                kw_vector = dictionary.doc2bow(view_product)
                                # Đem đi tính toán cho ra dự đoán (similarity calculation)
                                sim = index[model_gensim[kw_vector]]
                                product_clean['gensim_similarity_score'] = sim
                                # Tìm kiếm các chỉ số này trong data_product dataframe, và trả về đúng số dòng = number_re
                                gensim_top_recommendations = product_clean
                                # Lọc lại kết quả nếu có yêu cầu
                                if threshold > 0:
                                    gensim_top_recommendations = gensim_top_recommendations.loc[gensim_top_recommendations["gensim_similarity_score"] >= threshold]
                                return gensim_top_recommendations.sort_values(by=['gensim_similarity_score'], ascending= False).head(number_re)
                            st.write('-----'*30)
                            gensim_start_time = datetime.now()
                            gensim_top_recommendations = gensim_recommendation(search_content, num_rows, 0.0)
                            st.write('Total Gensim run time : \t\t\t', datetime.now() - gensim_start_time)
                            st.write('Search Results by Gensim')
                            # st.dataframe(gensim_top_recommendations[['item_id','product_name','gensim_similarity_score']].style.applymap(lambda x: 'background-color: lightgreen; color: black'))
                            st.dataframe(gensim_top_recommendations[['item_id','product_name','gensim_similarity_score']].style.applymap(lambda x: 'background-color: lightgreen; color: black'))
                        if search_by_cosine_similarity:
                            #1. Create
                            STOP_WORD_FILE = 'vietnamese-stopwords.txt'
                            with open(f'Project_{project_num}/Data/' + STOP_WORD_FILE, 'r', encoding='utf-8') as file:
                                stop_words = file.read()
                            stop_words = stop_words.split('\n')
                            tf = TfidfVectorizer(analyzer='word', min_df=0.0, stop_words=stop_words)
                            # Lưu ý: description_wt phải được làm toàn bộ chỉnh chu vào
                            tfidf_matrix = tf.fit_transform(product_clean.description_wt)
                            def cosin_recommendation(view_product_name, number_re= 3, threshold= 0):
                                global tfidf_matrix
                                if len(view_product_name) > 255:
                                    view_product_name = view_product_name[:255]
                                # Transform the search string into tf-idf vector
                                search_tfidf = tf.transform([view_product_name])

                                # Calculate the cosine similarities with the existing tf-idf matrix
                                cosine_similarities = cosine_similarity(search_tfidf, tfidf_matrix)

                                # Add the cosine similarity values as a new column in the dataframe
                                product_clean['cosin_similarities_score'] = cosine_similarities[0]

                                # Sort the dataframe by cosine similarity in descending order and get the top products
                                cosin_top_recommendations = product_clean

                                # Filter the results if a threshold is specified
                                if threshold > 0:
                                    cosin_top_recommendations = cosin_top_recommendations.loc[cosin_top_recommendations["cosin_similarities_score"] >= threshold]
                                return cosin_top_recommendations.sort_values(by=['cosin_similarities_score'], ascending=False).head(number_re)
                            st.write('-----'*30)
                            cosin_start_time = datetime.now()
                            cosin_top_recommendations  = cosin_recommendation(search_content, num_rows, 0.0)
                            st.write('Total cosine_similarity run time : \t\t', datetime.now() - cosin_start_time)
                            st.write('Search Results by Cosine Similarity')
                            # st.dataframe(cosin_top_recommendations[['item_id','product_name','gensim_similarity_score','cosin_similarities_score']].style.applymap(lambda x: 'background-color: lavender; color: black'))
                            st.dataframe(cosin_top_recommendations[['item_id','product_name','gensim_similarity_score','cosin_similarities_score']].style.applymap(lambda x: 'background-color: lavender; color: black'))
                        if search_by_als:
                            st.write('Search Results by ALS')
                            # st.write(df_ALS)
                        # diff = pd.DataFrame((gensim_top_recommendations['product_name'].values == cosin_top_recommendations['product_name'].values))
                        

                        ## SO SÁNH CÁC GIẢI PHÁP
                        if search_by_cosine_similarity == True and search_by_gensim == True:
                            st.write('-----'*30)
                            st.write('**SO SÁNH KẾT QUẢ TRẢ VỀ CÁC GIẢI PHÁP**')
                            def highlight_diff(row):
                                if row['solution'] == 'gensim_only':
                                    return ['background-color: lightgreen; color: black']*len(row)
                                elif row['solution'] == 'consin_only':
                                    return ['background-color: lavender; color: black']*len(row)
                                else:
                                    return ['']*len(row)
                            # In ra các dòng có nội dung khác nhau
                            diff = pd.merge(gensim_top_recommendations[['product_name']], cosin_top_recommendations[['product_name']], on='product_name', how='outer', indicator=True)
                            diff = diff[diff['_merge'] != 'both'].rename(columns={'_merge':'solution'})
                            diff['solution'] = diff['solution'].replace({'left_only': 'gensim_only', 'right_only': 'consin_only'})
                            diff = diff.style.apply(highlight_diff, axis=1)
                            if diff.data.empty:
                                st.write("=> Hai giải pháp cho kết quả gợi ý như nhau.")
                            st.dataframe(diff)
                            st.write('Total run time Content-based filtering - Recommendation:',str(datetime.now() - t0_gensim_cosine_als))
                with tab2:
                    # Create 3 columns
                    col1, col2, col3 = st.columns(3)
                    # Create checkboxes for search options
                    search_by_gensim = col1.checkbox(label='Search by Gensim*',value=True)
                    search_by_cosine_similarity = col2.checkbox(label='Search by Cosine Similarity',value=True)
                    search_by_als = col3.checkbox(label='Search by ALS_', disabled=True)
                    import numpy as np
                    # Không cho Streamlit mỗi lần nhấn nút chọn sản phẩm bị load lại 9 sản phẩm mới ^^
                    if 'sampled_df' not in st.session_state:
                        st.session_state.sampled_df = product_clean.sample(9)
                    sampled_df = st.session_state.sampled_df
                    item_ids = sampled_df['item_id'].to_numpy()
                    item_ids = np.reshape(item_ids, (3, 3))
                    image_paths = [f'Project_{project_num}/images/product/product_1.png', f'Project_{project_num}/images/product/product_2.png', \
                                   f'Project_{project_num}/images/product/product_3.png', f'Project_{project_num}/images/product/product_4.png',\
                                   f'Project_{project_num}/images/product/product_5.png',f'Project_{project_num}/images/product/product_6.png', \
                                   f'Project_{project_num}/images/product/product_7.png',f'Project_{project_num}/images/product/product_8.png',f'Project_{project_num}/images/product/product_9.png']
                    for i in range(3):
                        cols = st.columns(3)
                        for j in range(3):
                            item = sampled_df.iloc[i*3+j]
                            with cols[j]:
                                st.image(image_paths[i*3+j], caption=item['brand'])
                                placeholder = st.empty()
                                if placeholder.button(f'Item {i*3+j+1}', key=f'button_{i*3+j}'):
                                    product_code_currently_viewing = item["item_id"]
                                    st.dataframe(product_clean.loc[product_clean['item_id'] == product_code_currently_viewing,['item_id','product_name','group','brand','description_wt']])
                                    # Display the search results based on the selected options
                                    if search_by_gensim:
                                        ## Gensim
                                        #1. Load Gensim model
                                        # Tải mô hình và từ điển
                                        model_gensim = gensim.models.TfidfModel.load(f'Project_{project_num}/gensim_models/tfidf.model')
                                        # Đọc từ điển đã được lưu trữ lên dùng
                                        dictionary = gensim.corpora.Dictionary.load(f'Project_{project_num}/gensim_models/dictionary.gensim')
                                        # Đọc index đã được lưu trữ lên dùng
                                        index = gensim.similarities.Similarity.load(f'Project_{project_num}/gensim_models/index_file')
                                        def gensim_recommendation(view_product_name, number_re= 3, threshold= 0):
                                            '''
                                            view_product_name:   sản phẩm đang cần tìm
                                            number_re:              trả về bao nhiêu kết quả tìm được tốt nhất
                                            threshold:              lọc lại kết quả có điểm >= similarity_score mới trả về
                                            '''
                                            if len(view_product_name) > 255:
                                                view_product_name = view_product_name[:255]
                                            # Rả ra thông tin
                                            view_product = view_product_name.lower().split()
                                            # Chuyển đổi các từ tìm kiếm thành Vectơ thưa thớt (Convert search words into Sparse Vectors)
                                            kw_vector = dictionary.doc2bow(view_product)
                                            # Đem đi tính toán cho ra dự đoán (similarity calculation)
                                            sim = index[model_gensim[kw_vector]]
                                            product_clean['gensim_similarity_score'] = sim
                                            # Tìm kiếm các chỉ số này trong data_product dataframe, và trả về đúng số dòng = number_re
                                            gensim_top_recommendations = product_clean
                                            # Lọc lại kết quả nếu có yêu cầu
                                            if threshold > 0:
                                                gensim_top_recommendations = gensim_top_recommendations.loc[gensim_top_recommendations["gensim_similarity_score"] >= threshold]
                                            return gensim_top_recommendations.sort_values(by=['gensim_similarity_score'], ascending= False).head(number_re)
                                        st.write('-----'*30)
                                        view_product_name = None
                                        try:
                                            # view_product_name = get_product_name(product_code_currently_viewing)['description_wt'].values[0]
                                            view_product_name = get_product_name(product_code_currently_viewing)['product_name'].values[0]
                                        except:
                                            print("\n+ Please double-check product_id", product_code_currently_viewing," not existed !!!")
                                        else:
                                            gensim_start_time = datetime.now()
                                            gensim_top_recommendations = gensim_recommendation(view_product_name, num_rows, 0.0)
                                            st.write('Total Gensim run time : \t\t\t', datetime.now() - gensim_start_time)
                                            st.write('Search Results by Gensim')
                                            # st.dataframe(gensim_top_recommendations[['item_id','product_name','gensim_similarity_score']].style.applymap(lambda x: 'background-color: lightgreen; color: black'))
                                            st.dataframe(gensim_top_recommendations[['item_id','product_name','gensim_similarity_score']].style.applymap(lambda x: 'background-color: lightgreen; color: black'))
                                    if search_by_cosine_similarity:
                                        #1. Create
                                        STOP_WORD_FILE = 'vietnamese-stopwords.txt'
                                        with open(f'Project_{project_num}/Data/' + STOP_WORD_FILE, 'r', encoding='utf-8') as file:
                                            stop_words = file.read()
                                        stop_words = stop_words.split('\n')
                                        tf = TfidfVectorizer(analyzer='word', min_df=0.0, stop_words=stop_words)
                                        # Lưu ý: description_wt phải được làm toàn bộ chỉnh chu vào
                                        tfidf_matrix = tf.fit_transform(product_clean.description_wt)
                                        def cosin_recommendation(view_product_name, number_re= 3, threshold= 0):
                                            global tfidf_matrix
                                            if len(view_product_name) > 255:
                                                view_product_name = view_product_name[:255]
                                            # Transform the search string into tf-idf vector
                                            search_tfidf = tf.transform([view_product_name])

                                            # Calculate the cosine similarities with the existing tf-idf matrix
                                            cosine_similarities = cosine_similarity(search_tfidf, tfidf_matrix)

                                            # Add the cosine similarity values as a new column in the dataframe
                                            product_clean['cosin_similarities_score'] = cosine_similarities[0]

                                            # Sort the dataframe by cosine similarity in descending order and get the top products
                                            cosin_top_recommendations = product_clean

                                            # Filter the results if a threshold is specified
                                            if threshold > 0:
                                                cosin_top_recommendations = cosin_top_recommendations.loc[cosin_top_recommendations["cosin_similarities_score"] >= threshold]
                                            return cosin_top_recommendations.sort_values(by=['cosin_similarities_score'], ascending=False).head(number_re)
                                        

                                        st.write('-----'*30)
                                        view_product_name = None
                                        try:
                                            # view_product_name = get_product_name(product_code_currently_viewing)['description_wt'].values[0]
                                            view_product_name = get_product_name(product_code_currently_viewing)['product_name'].values[0]
                                        except:
                                            print("\n+ Please double-check product_id", product_code_currently_viewing," not existed !!!")
                                        else:
                                            cosin_start_time = datetime.now()
                                            cosin_top_recommendations  = cosin_recommendation(view_product_name, num_rows, 0.0)
                                            st.write('Total cosine_similarity run time : \t\t', datetime.now() - cosin_start_time)
                                            st.write('Search Results by Cosine Similarity')
                                            st.dataframe(cosin_top_recommendations[['item_id','product_name','gensim_similarity_score','cosin_similarities_score']].style.applymap(lambda x: 'background-color: lavender; color: black'))
                                    if search_by_als:
                                        st.write('Search Results by ALS')
                                        # st.write(df_ALS)
                                    # diff = pd.DataFrame((gensim_top_recommendations['product_name'].values == cosin_top_recommendations['product_name'].values))
                                    

                                    ## SO SÁNH CÁC GIẢI PHÁP
                                    if search_by_cosine_similarity == True and search_by_gensim == True:
                                        st.write('-----'*30)
                                        st.write('**SO SÁNH KẾT QUẢ TRẢ VỀ CÁC GIẢI PHÁP**')
                                        def highlight_diff(row):
                                            if row['solution'] == 'gensim_only':
                                                return ['background-color: lightgreen; color: black']*len(row)
                                            elif row['solution'] == 'consin_only':
                                                return ['background-color: lavender; color: black']*len(row)
                                            else:
                                                return ['']*len(row)
                                        # In ra các dòng có nội dung khác nhau
                                        diff = pd.merge(gensim_top_recommendations[['product_name']], cosin_top_recommendations[['product_name']], on='product_name', how='outer', indicator=True)
                                        diff = diff[diff['_merge'] != 'both'].rename(columns={'_merge':'solution'})
                                        diff['solution'] = diff['solution'].replace({'left_only': 'gensim_only', 'right_only': 'consin_only'})
                                        diff = diff.style.apply(highlight_diff, axis=1)
                                        if diff.data.empty:
                                            st.write("=> Hai giải pháp cho kết quả gợi ý như nhau.")
                                        st.dataframe(diff)
                                        st.write('Total run time Content-based filtering - Recommendation:',str(datetime.now() - t0_gensim_cosine_als))
                
            st.write('Total run time function Recommendation:',str(datetime.now() - t0_recommendation))
                        
    else:
        print("Do something")
    # st.write(f'Total run time Project_{project_num} is: {str(datetime.now() - first_step)}')
else:
    # Hiển thị tên của dự án 
    st.subheader("Project 3: Sentiment Analysis")
    # Hiển thị danh sách các tùy chọn cho người dùng lựa chọn từng bước trong dự án
    step = left_column.radio('', ['Business Understanding', 'Preprocessing + EDA', 'Applicable models', 'Prediction'])
    # Xử lý sự kiện khi người dùng lựa chọn từng mục trong danh sách và hiển thị hình ảnh tương ứng
    if step == 'Business Understanding':
        st.image(f'Project_{project_num}/images/a{project_num}.jpg')
    else:
        print("Do something")
