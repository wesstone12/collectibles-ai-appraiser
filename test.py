import datetime
from ebaysdk.finding import Connection
api = Connection(config_file='config.yaml')
response = api.execute('findItemsAdvanced', {'keywords': 'pokemon pikachu ex'})

print(response.reply)
assert(response.reply.ack=='Success')  
assert(type(response.reply.searchResult.item)==list)

item = response.reply.searchResult.item[0]
assert(type(item.listingInfo.endTime)==datetime.datetime)