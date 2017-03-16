from django.db import models
import datetime
from django.utils import timezone

# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.


class Messages(models.Model):
    id = models.IntegerField(db_column='Id', primary_key=True, blank=True)
    origin = models.TextField(db_column='Origin', blank=True, null=True)
    datetime = models.DateTimeField(db_column='DateTime', blank=True, null=True)
    user = models.CharField(max_length=200, db_column='User', blank=True, null=True)
    content = models.TextField(db_column='Content', blank=True, null=True)
    block = models.IntegerField(db_column='Block', blank=True)

    class Meta:
        db_table = 'Messages'

    def __str__(self):
        return '[{date}] {user} : {message}'.format(date=self.datetime,
                                                    user=self.user, message=self.content)
