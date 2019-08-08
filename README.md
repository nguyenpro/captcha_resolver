# captcha_resolver
1.	CÀI PYTHON 3.7: https://tecadmin.net/install-python-3-7-on-centos/
2.	CÀI thư viện opencv-python bằng pip:
Gõ lệnh: python3.7 –m  pip install opencv-python
3.	CÀI LibSM:
Gõ lệnh: sudo yum install libXext libSM libXrender

#Chay local
python3.7 captcha_local.py --image <img_path>


#Chay http server
4.	Mở port 9000:
step 1
	firewall-cmd --zone=public --permanent --add-port=9000/tcp
Step 2
	firewall-cmd –reload

5.	Chạy bằng pm2 để server tự restart khi bị stop, chạy bằng lệnh:
     pm2 start captcha_http.py
Nếu muốn chạy với nhìu process bằng pm2, gõ lệnh:
     pm2 start captcha_http.py –i 4
Nghĩa là chạy 4 process

6. Hướng dẫn cài pm2: https://computingforgeeks.com/install-pm2-node-js-process-manager-on-rhel-centos-8/
