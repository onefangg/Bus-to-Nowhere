import telegram
from telegram.ext import CommandHandler,ConversationHandler, MessageHandler, Updater, Filters
import logging
from telegram import ReplyKeyboardMarkup, Update, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
import pandas as pd
import numpy as np
from math import cos, asin, sqrt, pi
import os
import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import folium as fm
import webbrowser
import osmnx as ox
import networkx as nx
import io
from PIL import Image

PORT = int(os.environ.get('PORT', 5000))
TOKEN = '' #FOR DEPLOYMENT
bot = telegram.Bot(token=TOKEN)
print(bot.get_me())

##For error handling 
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)
logger = logging.getLogger()

##Base functions for bot
def start(update, context):
    global stops
    global routes
    global ID
    ID = update.message.chat.id
    reply_keyboard = [[KeyboardButton("Send current location 📍", request_location=True)   ]]
    bot.send_message(ID,
        "Welcome to the bus journey into the unknown!\n"
        "Want to go on a 🚌 bus journey 🚌 but don't know where to go? I am here to help! 😊\n\n"
    )
    update.message.reply_text(
        "Send me your current location to start!",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),
    )
    return LOCATION

def help(update, context):
    update.message.reply_text("Type /start to start the bot, or /done to restart the conversation!")

def unknown(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")


#Here are the telegram functions

LOCATION, DISTANCE, TRAVEL, MORE = range(4)
user_location = {}
def get_location(update, context):
    curr_loc = update.message.location
    if curr_loc is None:
        reply_keyboard = [[KeyboardButton("Send current location 📍", request_location=True)   ]]
        update.message.reply_text(
        "Send me your current location to start",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True),
        )
        return LOCATION
    global LAT, LONG
    longitude = curr_loc.longitude
    latitude = curr_loc.latitude
    LAT = latitude
    LONG = longitude
    user_location['longitude'] = longitude
    user_location['latitude'] = latitude
    locator = Nominatim(user_agent="Openstreetmap")
    coordinates = str(latitude) + ", " + str(longitude)
    location = locator.reverse(coordinates)
    loc_name = str(location.raw['display_name'])
    reply_keyboard = [[KeyboardButton('Yes')],[ KeyboardButton('No, I would like to send another location')]]
    update.message.reply_text(
        "We have identified your current location as: " + loc_name +
        "\nIs this location correct?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),) 
    return DISTANCE

def get_distance(update, context):
    prev = update.message.text
    if prev == "Yes":
        reply_keyboard = [[KeyboardButton('5 - 15 KM'), KeyboardButton('15 - 25 KM'), KeyboardButton('25 - 35 KM')],
                          [ KeyboardButton('35 - 45 KM'), KeyboardButton('Longer than 45 KM')]]
        update.message.reply_text(
            "How far would you like to travel?",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
        )
        return TRAVEL
    elif prev.lower().find("no") > -1:
        update.message.reply_text(
            "Send another location by tapping the paper clip icon 📎 > Location",
        )
        return LOCATION
    else:
        return DISTANCE

def send_bus(update, context):
    global routes_df, counter, counter_limit, bus_stop_count,bus_stop_count_limit, rt, all_buses, img_counter, upper, lower, codes,start1
    chosen_distance = update.message.text
    possible_dist = ['5 - 15 KM', '15 - 25 KM', '25 - 35 KM','35 - 45 KM', 'Longer than 45 KM']
    if chosen_distance not in possible_dist:
        reply_keyboard = [[KeyboardButton('5 - 15 KM'), KeyboardButton('15 - 25 KM'), KeyboardButton('25 - 35 KM')],
                          [ KeyboardButton('35 - 45 KM'), KeyboardButton('Longer than 45 KM')]]
        update.message.reply_text(
            "How far would you like to travel?",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),
        )
        return TRAVEL
    bot.send_message(ID, "You have chosen: " + chosen_distance + "\nFetching bus journey...")
    if chosen_distance == 'Longer than 45 KM':
        lower = 45
        upper = 75
    else:
        lower = chosen_distance[0:chosen_distance.find(" ")]
        upper = chosen_distance[chosen_distance.find("-")+2:chosen_distance.find("-")+2+2]
    global bus_stops_dic, img_in, img_out
    bus_stops_dic = get_bus_stops(user_location['latitude'],user_location['longitude'])    
    routes_df = [] 
    img_in = './route_img/img_{}.html'
    img_out = './route_img/img_{}.jpg'
    counter = 0
    img_counter = 0
    codes = []
    for code, details in bus_stops_dic.items():
        codes.append(code)
    bus_stop_count_limit = len(codes)
    bus_stop_count = 0
    all_buses = bus_stops_dic[codes[bus_stop_count]][1]['ServiceNo'].values.tolist() #list of buses from this stop
    counter_limit = len(all_buses)
    start1 = str(codes[bus_stop_count])
    lower = int(lower)
    upper = int(upper)
    rt = fetch_routes(all_buses[counter], start1, lower, upper)
    bus = all_buses[counter]
    try:
        res = plot_route(rt)
        img_map, attn_list = res[0], res[1]
        img_map.save(img_in.format(img_counter))
        
        start_info = df[df['BusStopCode'] == int(start1)]['Description'].iloc[0]
        end = rt['BusStopCode'].iloc[-1]
        end_info = df[df['BusStopCode'] == int(end)]['Description'].iloc[0]
        bot.send_message(ID, "Head to " + str(start_info)+ ", take bus " + str(bus) + " to start your journey which will end at " + str(end_info))
        html_file = open(img_in.format(img_counter), 'rb')
        bot.sendDocument(ID, html_file)
        img_counter += 1
    except nx.exception.NetworkXNoPath:
        pass    
    reply_keyboard = [[KeyboardButton('Yes')],[ KeyboardButton('No, give me another')]]
    update.message.reply_text(
        "Are you happy with the suggested bus journey?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),) 
    return MORE

def more_routes(update,context):
    global routes_df, counter, counter_limit, bus_stop_count,bus_stop_count_limit, rt, all_buses, img_counter, upper, lower, codes,start1
    response = update.message.text.lower()
    if response.find("no") > -1:
        counter += 1
        if counter >= counter_limit:
            bus_stop_count += 1
            if bus_stop_count >= bus_stop_count_limit:
                bot.send_message(ID, "Tough, that's all we have for you.")
                return done(update, context)
            else:
                bot.send_message(ID, "Okay, we'll get you another journey....")
                counter = 0
                all_buses = bus_stops_dic[codes[bus_stop_count]][1].values.tolist()[0]
                counter_limit = len(all_buses)
                start1 = str(codes[bus_stop_count])
                lower = int(lower)
                upper = int(upper)
                rt = fetch_routes(all_buses[counter], start1, lower, upper)
                
                bus = all_buses[counter]
                try:
                    res = plot_route(rt)
                    img_map, attn_list = res[0], res[1]
                    img_map.save(img_in.format(img_counter))
                    
                    start_info = df[df['BusStopCode'] == int(start1)]['Description'].iloc[0]
                    end = rt['BusStopCode'].iloc[-1]
                    end_info = df[df['BusStopCode'] == int(end)]['Description'].iloc[0]
                    bot.send_message(ID, "Head to " + str(start_info)+ ", take bus " + str(bus) + " to start your journey which will end at " + str(end_info))
                    html_file = open(img_in.format(img_counter), 'rb') #Sub this for the actual picture
                    bot.sendDocument(ID, html_file)
                    img_counter += 1
                except nx.exception.NetworkXNoPath:
                    pass
                return MORE
        else:
            lower = int(lower)
            upper = int(upper)
            bot.send_message(ID, "Okay, we'll get you another journey...")
            rt = fetch_routes(all_buses[counter], start1, lower, upper)
            bus = all_buses[counter]
            try:
                res = plot_route(rt)
                img_map, attn_list = res[0], res[1]
                img_map.save(img_in.format(img_counter))
                
                start_info = df[df['BusStopCode'] == int(start1)]['Description'].iloc[0]
                end = rt['BusStopCode'].iloc[-1]
                end_info = df[df['BusStopCode'] == int(end)]['Description'].iloc[0]
                bot.send_message(ID, "Head to " + str(start_info)+ ", take bus " + str(bus) + " to start your journey which will end at " + str(end_info))
                html_file = open(img_in.format(img_counter), 'rb') #Sub this for the actual picture
                bot.sendDocument(ID, html_file)
                img_counter += 1
            except nx.exception.NetworkXNoPath:
                pass
            return MORE
        
    elif response.find("yes") > -1:
        bot.send_message(ID, "Fantastic! Hope you enjoy your bus journey.")
        return done(update, context)
    else:
        reply_keyboard = [[KeyboardButton('Yes')],[ KeyboardButton('No, give me another')]]
        update.message.reply_text(
            "Are you happy with the suggested bus journey?",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True),) 
        return MORE  

#Helper functions
def distance(row, lat1, lon1):
    #This function was adapted from https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude
    lat2 = row['Latitude']
    lon2 = row['Longitude']
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))

def get_bus_stops(user_lat, user_long): 
    global df
    df = pd.read_csv("bus_stops.csv")
    df['distance'] = df.apply(lambda row: distance(row, user_lat, user_long), axis=1)
    df2 = df.nsmallest(2, ['distance'])
    code1 = df2['BusStopCode'].iloc[0]
    code2 = df2['BusStopCode'].iloc[1]
    if df2['distance'].iloc[1] > 1:
        return check_overlap(code1, False, df2.iloc[0])
    else:
        return check_overlap(code1, code2, df2) 
    
def check_overlap(code1, code2, dataframe):
    route1 = routes[routes['BusStopCode'] == str(code1)]    
    if code2 == False:
        return {code1: [dataframe, route1]}    
    route2 = routes[routes['BusStopCode'] == str(code2)]
    s1 = pd.merge(route1, route2, how='inner', on=['ServiceNo'])
    if s1.empty:
        return {code1: [dataframe.iloc[0], route1], code2: [dataframe.iloc[1], route2]}
    s1['direction overlap'] = s1.apply(lambda x: "Yes" if x['Direction_x'] == x['Direction_y'] else "No", axis = 1)
    overlap = "Yes" in s1['direction overlap'].unique()
    if overlap:
        #check for most routes: 
        len1 = len(route1.index) 
        len2 = len(route2.index)
        if len1 >= len2:
            return {code1: [dataframe.iloc[0], route1]} 
        else:
            return {code2: [dataframe.iloc[1], route2]}
    else: 
        return {code1: [dataframe.iloc[0], route1], code2: [dataframe.iloc[1], route2]}

    
landmarks = pd.read_csv("Malls and MRT Stations.csv")
routes = pd.read_csv("bus_routes.csv")
routes = routes.drop(columns = routes.columns.values[6:])
stops = pd.read_csv("cleaned_bus_stops.csv")
stops = stops.fillna('')

def preprocess_df(df):
    key=list(df['BusStopCode'])
    val=list(zip(df['Latitude'], df['Longitude'], df['Attraction']))
    return dict(zip(key,val))

stops = preprocess_df(stops)
def plot_route(rt):
    def prep_route(g, nodes):
        res = []
        for i in range(len(nodes)-1):
            u = nodes[i]
            v = nodes[i+1]
            if g.get_edge_data(u,v):
                res.append(v)
            else:
                res.extend(nx.shortest_path(g, u, v)[1:])
        return res

    dis = fm.Map(location=[LAT, LONG], zoom_start=15)
    
    all_stops = rt.BusStopCode.astype(np.int64).values
    points = []
    attn = []
    print('plot_route')

    first_stop = stops[all_stops[0]]
    points.append(first_stop[:2])
    fm.Marker(first_stop[:2], icon = fm.Icon(color = 'green')).add_to(dis)

    for s in all_stops[1:-1]:
        points.append(stops[s][:2])
        if stops[s][2]:
            print('found attn')
            attn.append(stops[s][2])
            fm.Marker(stops[s][:2], icon = fm.Icon(color = 'purple')).add_to(dis)
        else:
            fm.Marker(stops[s][:2]).add_to(dis)

    last_stop = stops[all_stops[-1]]
    points.append(last_stop[:2])
    fm.Marker(last_stop[:2], icon = fm.Icon(color = 'red')).add_to(dis)
    lat = [x[0] for x in points]
    long = [x[1] for x in points]

    n,s,e,w = max(lat)+0.005, min(lat)-0.005,max(long)+0.005, min(long)-0.005
    graph = ox.graph.graph_from_bbox(n,s,e,w,
                                     network_type = 'drive_service')
    nodes = [ox.get_nearest_node(graph, pt) for pt in points]
    
    rte = prep_route(graph, nodes)
    return ox.folium.plot_route_folium(graph, rte, dis, route_color = "#1E90FF"), attn

def fetch_routes(bus, stop, *args):
    # args: any additional constraints
    base = routes[routes.ServiceNo == bus].reset_index()
    idx = base[base.BusStopCode == stop].index.values[0]
    sbst = base.iloc[idx:,].copy()
    try:
        brk_idx = sbst[sbst.Direction==2].index.values[0]
        sbst = base.iloc[idx:brk_idx,][['BusStopCode', 'Distance']].copy()
    except IndexError:
        pass
    constraints = list(args)
    lower = constraints[0]
    upper = constraints[1]
    dist = sbst.Distance.diff().cumsum().fillna(0)
    bus_idx = dist.loc[lambda x: x <= upper].index.values
    query = base.iloc[bus_idx,5:7]
    return query 



def done(update, context):
    bot.send_message(ID,
        "Thank you for using Bus to Nowhere! \nType /start to start another journey."
        )
    return ConversationHandler.END

def main() -> None:
    updater = Updater(token=TOKEN,use_context=True)
    dispatcher = updater.dispatcher
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            LOCATION: [
                MessageHandler(
                    Filters.location | Filters.text, get_location
                ),CommandHandler('done', done)
            ],
            DISTANCE: [
                MessageHandler(
                    (Filters.text)& ~(Filters.command | Filters.regex('^done$')), get_distance
                ),CommandHandler('done', done)
            ],
            TRAVEL: [
                MessageHandler(
                    (Filters.text)& ~(Filters.command | Filters.regex('^done$')), send_bus
                ),CommandHandler('done', done)
            ],
            MORE: [
                MessageHandler(
                    (Filters.text)& ~(Filters.command | Filters.regex('^done$')), more_routes
                ),CommandHandler('done', done)
            ],
        },
        fallbacks=[CommandHandler('done', done)],
    )
    dispatcher.add_handler(conv_handler)

    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('done', done))
    dispatcher.add_handler(CommandHandler('help', help))
    dispatcher.add_handler(MessageHandler(Filters.command, unknown))
    
    # Start the Bot
##    updater.start_polling() #FOR TESTING

    #FOR DEPLOYMENT
    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN)
    updater.bot.setWebhook('https://bus-to-nowhere.herokuapp.com/' + TOKEN)
    updater.idle()


if __name__ == '__main__':
    main()
