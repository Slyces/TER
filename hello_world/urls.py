from django.conf.urls import url
from . import views

app_name = "hello_world"

urlpatterns = [
    url(r'^$', views.index.as_view(), name='index'),
    url(r'^[Ii]ntroduction/?$', views.introduction.as_view(), name="Introduction"),
    url(r'^[Cc]oncepts?/?$', views.concepts.as_view(), name="Concepts"),
    url(r'^[Ff]inance/?$', views.finance.as_view(), name="Finance"),
    url(r'^[Aa]pplication/?$', views.application.as_view(), name="Application"),
    url(r'^[Cc]onclusion/?$', views.conclusion.as_view(), name="Conclusion"),
    url(r'^[Rr]esulta?t?s?/?$', views.ResultsView.as_view(), name="Resultats"),
    url(r'^test$', views.test.as_view(), name="Test"),
]
