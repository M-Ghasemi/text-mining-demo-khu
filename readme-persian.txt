راه اندازی و اجرای پروژه:
فایل اصلی هم به صورت اسکریپت پایتون، و هم به صورت فایل jupyter notebook است.
جهت اجرای پروژه نیاز است که پکیج های لیست شده در فایل requirements.txt بر روی پایتون ۳ نصب شده باشند.


نصب کتابخانه‌های مورد نیاز
در صورت نصب بسته  Anaconda می‌توان از دستور زیر برای نصب تمام کتابخانه‌های مورد نیاز اقدام کرد(در مک و لینوکس):
while read requirement; do conda install --yes $requirement; done < requirements.txt
در غیر این صورت برای نصب با pip از دستور زیر برای نصب تمامی کتابخانه‌های مورد نظر استفاده کرد:
pip install -r requirements.txt --no-index --find-links
برای نصب کتابخانه‌ها بصورت جدا هم نیز می‌توان از دستورهای زیر برای pip و anacoda استفاده کرد:
pip install 'package name'
conda install 'package name'

نصب داده‌های NLTK
برای اجرای قسمت‌های پیش‌پردازش متن باید در ابتدا داده‌های nltk نصب شود:
برای نصب داده‌های nltk باید در محیط پایتون دستورهای زیر اجرا شوند:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
از دستور زیر برای نصب تمامی داده‌های nltk می‌توان استفاده کرد:
nltk.download('all')

اجرای پروژه

علاوه بر پکیج های موجود در requirements.txt برای اجرای فایل نوت بوک، به نصب jupyter notebook نیز احتیاج است.

نصب jupyer notebook
برای نصب jupyter توصیه می‌شود ابتدا بسته Anaconda نصب شود.
Anaconda: https://www.anaconda.com/download/
Jupyter: https://jupyter.org/install
در صورت عدم نصب Anaconda می‌توان jupyter را از طریق pip در پایتون نیز نصب کرد.
برای پایتون ۳
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
اجرای jupyer notebook
بعد از نصب jupyter برای اجرا نیاز هست تا دستور زیر در Terminal (مک و لینوکس) و یا command prompt (ویندوز) اجرا شود:
jupyter notebook

بعد باز کردن فایل پروژه در jupyter برای اجرا هر قسمت نیاز هست تا از گزینه run برای اجرا هر قطعه کد بهره برد و راه دیگر برای اجرای هر قطعه اینکه بروی گزینه run cells از طریق زبانه cell کلیک شود.
