with open('/home/tuanminh/mir_project/src/mir_robot/tm/testpc.py', 'r') as f:
    text = f.read()

text = text.replace("if __name__ == '__main__':", """def main():
    app = QApplication(sys.argv)
    window = TestPCApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':""")

with open('/home/tuanminh/mir_project/src/mir_robot/tm/testpc.py', 'w') as f:
    f.write(text)
