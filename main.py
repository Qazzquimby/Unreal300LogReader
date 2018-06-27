import abc
import itertools
import math
import re
from datetime import datetime

import numpy as np
import pandas


# import matplotlib.pyplot as plt

# region Heroes


class Hero(object):
    curr_index = 0

    def __init__(self, name):
        self.name = name
        self.index = Hero.curr_index
        Hero.curr_index += 1

    def __str__(self):
        return self.name


ANA = Hero("Ana")
BASTION = Hero("Bastion")
BRIGITTE = Hero("Brigitte")
DVA = Hero("DVA")
DOOMFIST = Hero("Doomfist")
GENJI = Hero("Genji")
HANZO = Hero("Hanzo")
JUNKRAT = Hero("Junkrat")
LUCIO = Hero("Lucio")
MCCREE = Hero("McCree")
MEI = Hero("Mei")
MERCY = Hero("Mercy")
MOIRA = Hero("Moira")
ORISA = Hero("Orisa")
PHARAH = Hero("Pharah")
REAPER = Hero("Reaper")
REINHARDT = Hero("Reinhardt")
ROADHOG = Hero("Roadhog")
SOLDIER76 = Hero("Soldier76")
SOMBRA = Hero("Sombra")
SYMMETRA = Hero("Symmetra")
TORBJORN = Hero("Torbjorn")
TRACER = Hero("Tracer")
WIDOWMAKER = Hero("Widowmaker")
WINSTON = Hero("Winston")
ZARYA = Hero("Zarya")
ZENYATTA = Hero("Zenyatta")
HEROES = [ANA, BASTION, BRIGITTE, DVA, DOOMFIST, GENJI, HANZO, JUNKRAT, LUCIO, MCCREE, MEI, MERCY,
          MOIRA, ORISA, PHARAH,
          REAPER, REINHARDT, ROADHOG, SOLDIER76, SOMBRA, SYMMETRA, TORBJORN, TRACER, WIDOWMAKER,
          WINSTON, ZARYA,
          ZENYATTA]

# endregion

BLUE = "Blue"
RED = "Red"
TEAMS = [BLUE, RED]

# region Report Elements
class ReportElement(object):
    curr_index = 0

    def __init__(self, name, text):
        self.name = name
        self.text = text

        if text == "xxx":
            self.index = -1
        else:
            self.index = self.curr_index
            ReportElement.curr_index += 1

    def __str__(self):
        return self.name


START_TIME = ReportElement("Start", "Start: ")
END_TIME = ReportElement("End", "End: ")
LENGTH = ReportElement("Length", "xxx")
WINNING_TEAM = ReportElement("Winning team", "Winning team: ")
MAP = ReportElement("Map", "Map: ")
MODE = ReportElement("Mode index", "Mode index: ")
JOINS = ReportElement("Joins", "Joins: ")
LEAVES = ReportElement("Leaves", "Leaves: ")
BLUE_PLAYERS = ReportElement("Blue player count", "Blue player count: ")
RED_PLAYERS = ReportElement("Red player count", "Red player count: ")
# HERO_PLAY_RATES = ReportElement("Hero play rate", "{} {} Average count: ")
PLAYERS = ReportElement("Players", "xxx")
WIN_VALUES = ReportElement("Win value", "xxx")
LOSS_VALUES = ReportElement("Loss value", "xxx")

TEAM_HERO_START_POINT = ReportElement.curr_index

REPORT_ELEMENTS = [START_TIME, END_TIME, WINNING_TEAM, MAP, MODE, JOINS, LEAVES, BLUE_PLAYERS,
                   RED_PLAYERS]
for team in TEAMS:
    for hero in HEROES:
        team_hero_element = \
            ReportElement("{0}-{1}".format(team, hero.name),
                          "{0} {1} Average count: ".format(team, hero.name))
        REPORT_ELEMENTS.append(team_hero_element)

# endregion

ASSAULT_ESCORT = 0
ASSAULT = 1
CONTROL = 2
TDM = 3
ESCORT = 4
MODES = [ASSAULT_ESCORT, ASSAULT, CONTROL, TDM, ESCORT]
ASSYMETRIC_MODES = [ASSAULT_ESCORT, ASSAULT, ESCORT]

Z_CRITICAL = 1.65

GAMES_PLAYED = "Games Played"


class Main(object):
    def __init__(self):
        self.reports = self.build_reports()
        self.competitive_player_count = 5

        self.all_df, self.comp_df = self.build_dataframe()

        self.win_rate_tracker = WinRateTracker(self.comp_df)

        self.measure()
        pass

    def build_reports(self):
        lines = read_log()
        report_texts = [list(report) for i, report in
                        itertools.groupby(lines, lambda x: x == '=============================') if
                        not i]
        reports = []
        for report_text in report_texts:
            report = Report(report_text)
            reports.append(report)
        return reports

    def build_dataframe(self):
        lines = read_log()
        report_texts = [list(report) for i, report in
                        itertools.groupby(lines, lambda x: x == '=============================') if
                        not i]
        reports = []
        for report_text in report_texts:
            report = Report(report_text)
            array = report.to_array()
            reports.append(array)
        report_array = np.array(reports)

        column_names = [element for element in REPORT_ELEMENTS]

        dataframe = pandas.DataFrame(report_array, columns=column_names)

        # Filter player count
        is_competitive = dataframe[RED_PLAYERS] + dataframe[
            BLUE_PLAYERS] > self.competitive_player_count
        competitive_dataframe = dataframe[is_competitive]

        # Filter date to last major update
        # 6/14/2018 8:13:02 PM
        # Dva dm up. Torb dmg down, health up. Orisa health down. Payload speed up
        last_update = datetime.strptime("6/21/2018 12:54:00 am", "%m/%d/%Y %H:%M:%S %p")
        is_after_update = competitive_dataframe[START_TIME] > last_update
        competitive_dataframe = competitive_dataframe[is_after_update]

        return dataframe, competitive_dataframe

    def measure(self):
        total_hero_playtime = 0
        for team in TEAMS:
            for hero in HEROES:
                total_hero_playtime += \
                    self.comp_df[hero_team_report_element(hero, team)].sum()
        print("Total hero playtime {}".format(total_hero_playtime))
        print("Games {}".format(len(self.comp_df)))

        blue_winrate_AE = self.get_map_stats(ASSAULT_ESCORT)
        blue_winrate_A = self.get_map_stats(ASSAULT)
        blue_winrate_E = self.get_map_stats(ESCORT)

        hero_win_rates = {}
        for mode in MODES:
            hero_win_rates[mode] = {}
            for hero in HEROES:
                hero_win_rates[hero] = {}
                for direction in [-1, 1]:
                    hero_win_rates[mode][hero] = {}
                    for team in TEAMS:
                        hero_win_rates[mode][hero][team] = \
                            self.hero_winrate_in_mode_on_team(hero, mode, team, direction)

        hero_combined_win_rates = {}
        for hero in HEROES:
            hero_combined_win_rates[hero] = {}
            hero_combined_win_rates[hero][GAMES_PLAYED] = self.get_games_played(hero)
            for direction in [-1, 0, 1]:
                win_rate = self.hero_win_rate_total(hero, direction)
                hero_combined_win_rates[hero][direction] = win_rate
                CUTOFF = 0.52  # 0.5 is good too
                if (direction == -1 and win_rate > 0.0 + CUTOFF) \
                        or (direction == 1 and win_rate < 1.0 - CUTOFF):
                    print("Win rate for {0}, in direction {1}, is {2} "
                          .format(hero.name, direction, win_rate))

        play_rates = [(hero_combined_win_rates[hero][GAMES_PLAYED], hero) for hero in HEROES]
        play_rates.sort()

        self.get_win_rate_by_team_advantage()

        pass

    def get_map_stats(self, mode):
        map_stats = {}

        filtered_dataframe = self.comp_df[self.comp_df[MODE] == mode]
        map_stats[GAMES_PLAYED] = len(filtered_dataframe)

        for direction in [-1, 0, 1]:
            map_stats[direction] = self.blue_winrate_in_mode(mode, direction)

        return map_stats

    def blue_winrate_in_mode(self, mode, direction):
        #does NOT weight by team size advantage
        filtered_dataframe = self.comp_df[self.comp_df[MODE] == mode]

        blue_wins = filtered_dataframe[WINNING_TEAM] == BLUE
        win_values = filtered_dataframe[WIN_VALUES]
        weighted_blue_wins = blue_wins
        # weighted_blue_wins = weight_wins_by_advantage(blue_wins, win_values)

        if direction == 0:
            win_rate = np.average(weighted_blue_wins)
        else:
            win_rate = mean_confidence_interval(blue_wins, direction)[1]

        return win_rate

    def hero_win_rate_total(self, hero, direction):
        blue_wins = self.comp_df[WINNING_TEAM] == BLUE
        weighted_blue_wins = weight_wins_by_advantage(blue_wins, self.comp_df[WIN_VALUES])
        blue_weights = self.comp_df[hero_team_report_element(hero, BLUE)]

        red_wins = self.comp_df[WINNING_TEAM] == RED
        weighted_red_wins = weight_wins_by_advantage(red_wins, self.comp_df[WIN_VALUES])
        red_weights = self.comp_df[hero_team_report_element(hero, RED)]

        wins = np.concatenate((weighted_blue_wins, weighted_red_wins))
        weights = np.concatenate((blue_weights, red_weights))

        if len(wins) == 0:
            return 0.5

        if direction == 0:
            win_rate = np.average(wins, weights=weights)
        else:
            win_rate = weighted_mean_confidence_interval(wins, weights, direction)[1]

        return win_rate

    def hero_max_winrate_in_mode(self, hero, mode, direction):
        win_rate = max(self.hero_winrate_in_mode_on_team(hero, mode, BLUE, direction),
                       self.hero_winrate_in_mode_on_team(hero, mode, RED, direction))
        return win_rate

    def hero_winrate_in_mode(self, hero, mode, direction):
        blue_win_rate = self.hero_winrate_in_mode_on_team(hero, mode, BLUE, direction)
        red_win_rate = self.hero_winrate_in_mode_on_team(hero, mode, RED, direction)
        win_rate = (blue_win_rate + red_win_rate) / float(2)
        return win_rate

    def hero_winrate_in_mode_on_team(self, hero, mode, team, direction=-1):
        is_correct_mode = self.comp_df[MODE] == mode
        filtered_dataframe = self.comp_df[is_correct_mode]
        data = filtered_dataframe[WINNING_TEAM] == team

        # amount that hero was played
        weights = filtered_dataframe[hero_team_report_element(hero, team)]
        win_rate = weighted_mean_confidence_interval(data, weights, direction)[1]

        return win_rate

    def get_games_played(self, hero):
        blue_weights = self.comp_df[hero_team_report_element(hero, BLUE)]
        red_weights = self.comp_df[hero_team_report_element(hero, RED)]
        weights = np.concatenate((blue_weights, red_weights))
        total_play_time = sum(weights)
        return total_play_time



class Tracker(abc.ABC):
    def __init__(self, df):
        self.df = df


class WinRateTracker(Tracker):
    def __init__(self, df):
        super().__init__(df)

        self.get_effective_win_value()

    def get_effective_win_value(self):
        win_value_func = self.get_win_rate_by_team_advantage()

        win_values = []
        loss_values = []
        for row in self.df.iterrows():
            row = row[1]
            win_value = win_value_func(row)
            win_values.append(0.5 / win_value)
            loss_values.append(1 - 0.5 / win_value)
        win_values = pandas.Series(win_values)

        self.df.loc[:, WIN_VALUES] = pandas.Series(win_values).astype(float).values
        self.df.loc[:, LOSS_VALUES] = pandas.Series(loss_values).astype(float).values

    def get_win_rate_by_team_advantage(self):
        # blue_wins = (self.comp_df[WINNING_TEAM] == BLUE).astype(float)
        # blue_size_advantage \
        #     = (self.comp_df[BLUE_PLAYERS] / self.comp_df[RED_PLAYERS]).astype(float)
        #
        # blue_slope, blue_intercept, _, _, _ \
        #     = scipy.stats.linregress(blue_size_advantage, blue_wins)
        #
        # red_wins = (self.comp_df[WINNING_TEAM] == RED).astype(float)
        # red_size_advantage \
        #     = (self.comp_df[RED_PLAYERS] / self.comp_df[BLUE_PLAYERS]).astype(float)
        #
        # red_slope, red_intercept, _, _, _ \
        #     = scipy.stats.linregress(red_size_advantage, red_wins)
        #
        # slope = (blue_slope + red_slope) / 2
        # # intercept = (blue_intercept + red_intercept) / 2
        # intercept = 0.5 - slope  # fix so that average win rate will be 0.5

        blue_wins = (self.df[WINNING_TEAM] == BLUE).astype(float)
        blue_size_advantage \
            = (self.df[BLUE_PLAYERS] / self.df[RED_PLAYERS]).astype(float)

        # plt.plot(blue_size_advantage, blue_wins, "^b")
        #
        # coefs = np.polyfit(blue_size_advantage, blue_wins, 2)
        # results = []
        # x = np.linspace(0.5, 2, 10)
        # for x_i in x:
        #     sum = 0
        #     for i in range(len(coefs)):
        #         sum += coefs[len(coefs) - i - 1] * x_i ** i
        #     results.append(sum)
        # plt.plot(x, results)
        # plt.show()

        # coefs = [-0.45806834,  1.97327284, - 1.01742043]
        coefs = [- 1.01742043, 1.97327284, -0.45806834]

        def win_value(df_row):
            if df_row[WINNING_TEAM] == BLUE:
                winning_team_size_advantage = df_row[BLUE_PLAYERS] / df_row[RED_PLAYERS]
            elif df_row[WINNING_TEAM] == RED:
                winning_team_size_advantage = df_row[RED_PLAYERS] / df_row[BLUE_PLAYERS]
            else:
                assert False

            # win_amount = intercept + slope * winning_team_size_advantage
            win_amount = max(0, min(1,
                                    coefs[0]
                                    + coefs[1] * winning_team_size_advantage
                                    + coefs[2] * winning_team_size_advantage ** 2)
                             )
            return win_amount

        return win_value


class FunTracker(object):
    def __init__(self):
        pass


def hero_team_report_element(hero, team):
    if team == BLUE:
        return REPORT_ELEMENTS[TEAM_HERO_START_POINT + hero.index]
    elif team == RED:
        return REPORT_ELEMENTS[TEAM_HERO_START_POINT + hero.index + len(HEROES)]
    else:
        assert False


def weighted_mean_confidence_interval(data, weights, direction=-1):
    array = 1.0 * np.array(data)
    length = sum(weights)
    if length > 0:
        midpoint = np.average(array, weights=weights)

        variance = np.average((data - midpoint) ** 2, weights=weights)
        standard_error = math.sqrt(variance)

        # adistance = standard_error * sp.stats.t.pdf((1 + confidence) / 2., length)
        # z_critical = scipy.stats.norm.ppf(q=confidence)
        z_critical = Z_CRITICAL
        distance = z_critical * (standard_error / math.sqrt(length))

        return midpoint, midpoint + direction * distance
    else:
        return direction, 0


def weight_wins_by_advantage(won, win_value):
    # Rescale from 0to1 to -1to+1
    weighted_games = []
    for i, win in won.iteritems():
        if win == 0:
            weighted_game = 1 - win_value[i]
        elif win == 1:
            weighted_game = win_value[i]
        else:
            assert False
        weighted_games.append(weighted_game)

    weighted_games = pandas.Series(weighted_games).astype(float)
    return weighted_games


def mean_confidence_interval(data, direction=-1):
    array = 1.0 * np.array(data)
    length = len(array)
    if length > 0:
        midpoint = np.average(array)

        variance = np.average((data - midpoint) ** 2)
        standard_error = math.sqrt(variance)

        # distance = standard_error * sp.stats.t.pdf((1 + confidence) / 2., length)
        z_critical = Z_CRITICAL
        distance = z_critical * (standard_error / math.sqrt(length))

        return midpoint, midpoint + direction * distance
    else:
        return direction, 0


class Report(object):

    def __init__(self, text):
        self.text = text
        self._dict = {
            START_TIME: datetime.strptime(self.read(START_TIME), "%m/%d/%Y %H:%M:%S %p"),
            # 6/14/2018 8:13:02 PM
            END_TIME: datetime.strptime(self.read(END_TIME), "%m/%d/%Y %H:%M:%S %p"),
            LENGTH: datetime.strptime(self.read(END_TIME), "%m/%d/%Y %H:%M:%S %p")
                    - datetime.strptime(self.read(START_TIME), "%m/%d/%Y %H:%M:%S %p"),
            WINNING_TEAM: self.read(WINNING_TEAM),
            MAP: self.read(MAP),
            MODE: int(self.read(MODE)),
            JOINS: int(self.read(JOINS)),
            LEAVES: int(self.read(LEAVES)),
            BLUE_PLAYERS: float(self.read(BLUE_PLAYERS)),
            RED_PLAYERS: float(self.read(RED_PLAYERS)),
            PLAYERS: float(self.read(BLUE_PLAYERS)) + float(self.read(RED_PLAYERS)),
        }
        self.generate_hero_play_rates()

    def __getitem__(self, item):
        return self._dict[item]

    def read(self, element: ReportElement):
        line = self.text[element.index]
        value = line.split(element.text)[1]
        return value

    def generate_hero_play_rates(self):
        start_index = TEAM_HERO_START_POINT
        for team_i, team in enumerate(TEAMS):
            for hero in HEROES:
                element = hero_team_report_element(hero, team)
                play_rate_line = self.text[element.index]
                play_rate = max(0.0, float(play_rate_line.split(element.text)[1]))

                self._dict[element] = play_rate

    def to_array(self):
        elements = []
        for element in REPORT_ELEMENTS:
            elements.append(self[element])

        array = np.array(elements)
        return array


def read_log():
    with open(r"D:\life itself\Unreal300%\Logs\GameLogs.txt") as file:
        lines = file.readlines()
        cleaned_lines = clean_lines(lines)
        return cleaned_lines


def clean_lines(lines):
    new_lines = []
    for line in lines:
        line = re.sub('\n', '', line)
        if len(line) > 0:
            new_lines.append(line)
    return new_lines


if __name__ == '__main__':
    main = Main()


    # def get_competitive_reports(self):
    #     competitive_reports = [report for report in self.reports if
    #                            report[PLAYERS] >= self.competitive_player_count]
    #     return competitive_reports