#!/usr/bin/env python
import cvxpy as cp
from ortools.sat.python import cp_model
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QListWidget, QTableWidgetItem, QVBoxLayout, QWidget, QPushButton, QRadioButton, QGridLayout, QLabel, QComboBox, QDesktopWidget, QShortcut, QFrame
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.Qt import QButtonGroup
import sys
import time
from src.localization import Localization

Epsilon = 1e-6
Exact_Solution_Time = 0.01 #In seconds
N_Players = [3,4,5,6][3]
N_Cards = 21
Event_List = []#[[0, 0, 1], [0, 1, 1], [0, 2, 1], [0, 6, 1], [0, 7, 1], [0, 8, 1], [1, 3, 1], [2, 4, 1], [1, 9, 1], [2, 10, 1], [1, 12, 1], [1, 13, 1], [1, 14, 1], [2, 15, 1], [2, 16, 1], [2, 4, 10, 18, 0], [0, 0, 6, 19, 1]]
Unknown_Cards = []#player_idx, suspect_card, weapon_card, location_card
Known_Cards = []#player_idx, card_idx, probability (0/1)
Card_Counts = []
Redo_List = []
Card_Names = ['Professor Plum','Colonel Mustard','Mr. Green','Miss Scarlet','Dr. Orchid','Mrs. Peacock','Knife','Candlestick','Revolver','Lead Pipe','Rope','Wrench','Hall','Conservatory','Dining Room','Kitchen','Study','Library','Ballroom','Lounge','Billiard Room']
Suspect_Colors = [[255,0,255],[255,255,0],[0,255,0],[255,0,0],[255,255,255],[0,0,255]]

def Get_Screen_Size():
    desktop = QDesktopWidget()
    screen = desktop.screen()
    screen_size = screen.size()
    return screen_size

class VarArraySolutionCounter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = np.zeros((N_Cards,N_Players+1),dtype=int)
        self.__N_Solutions = 0
    def on_solution_callback(self):
        self.__N_Solutions += 1
        for i in range(N_Cards):
            for j in range(N_Players+1):
                self.__solution_count[i][j] += self.Value(self.__variables[i][j])
    def get_probabilities(self):
        return self.__solution_count.astype(np.float64)/self.__N_Solutions
    def get_total_solutions(self):
        return self.__N_Solutions

def Find_Probabilities_Exact():
    Start_Time = time.time()
    model = cp_model.CpModel()
    Card_Probabilities = [[model.NewBoolVar(name=str(i)+"_"+str(j)) for j in range(N_Players+1)] for i in range(N_Cards)]
    model.Add(sum([Card_Probabilities[i][-1] for i in range(0,6)])==1)
    model.Add(sum([Card_Probabilities[i][-1] for i in range(6,12)])==1)
    model.Add(sum([Card_Probabilities[i][-1] for i in range(12,N_Cards)])==1)
    for i in range(N_Cards):
        model.Add(sum([Card_Probabilities[i][j] for j in range(N_Players+1)])==1) #Every card belongs to someone or the solution
    for i in range(N_Players):
        model.Add(sum([Card_Probabilities[j][i] for j in range(N_Cards)])==Card_Counts[i])#Everyone has a certain number of cards
    for player_idx, card_idx, probability in Known_Cards:
        model.Add(Card_Probabilities[card_idx][player_idx]==probability)
    for i, u in enumerate(Unknown_Cards):
        player_idx, suspect_card, weapon_card, location_card = u
        model.Add(Card_Probabilities[suspect_card][player_idx]+Card_Probabilities[weapon_card][player_idx]+Card_Probabilities[location_card][player_idx]>=1)

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.max_time_in_seconds = Exact_Solution_Time
    solution_counter = VarArraySolutionCounter(Card_Probabilities)
    status = solver.Solve(model, solution_counter)
    if status == cp_model.OPTIMAL:
        print("Time:",time.time()-Start_Time,'Status = %s' % solver.StatusName(status),"N_Solutions",solution_counter.get_total_solutions())
        return solution_counter.get_probabilities()
    else:
        raise Exception()

def Find_Probabilities():
    try:
        return Find_Probabilities_Exact()
    except:
        pass
    Card_Probabilities = cp.Variable((N_Cards, N_Players+1))
    Card_Probabilities_Slack = cp.Variable((N_Cards, N_Players+1))
    if len(Unknown_Cards)>0:
        Unknown_Cards_Slack_Variables = cp.Variable((len(Unknown_Cards)))

    constraints = [Card_Probabilities>=0, #Probabilities are [0,1]
                   Card_Probabilities+Card_Probabilities_Slack==1, #Probabilities are [0,1]
                   Card_Probabilities_Slack>=0,
                   cp.sum(Card_Probabilities[0:6,-1],axis=0)==1, #There is a suspect card in the solution
                   cp.sum(Card_Probabilities[6:12,-1],axis=0)==1, #There is a weapon card in the solution
                   cp.sum(Card_Probabilities[12:N_Cards,-1],axis=0)==1, #There is a location card in the solution
                   cp.sum(Card_Probabilities,axis=1)==1, #Every card belongs to someone or the solution
                   cp.sum(Card_Probabilities[:,0:N_Players],axis=0)==Card_Counts, #Everyone has a certain number of cards
                   ]
    if len(Unknown_Cards)>0:
        constraints.append(Unknown_Cards_Slack_Variables>=0)

    for player_idx, card_idx, probability in Known_Cards:
        constraints.append(Card_Probabilities[card_idx,player_idx]==probability+Epsilon*(1 if probability==0 else -1))

    for i, u in enumerate(Unknown_Cards):
        player_idx, suspect_card, weapon_card, location_card = u
        constraints.append(Card_Probabilities[suspect_card,player_idx]+Card_Probabilities[weapon_card,player_idx]+Card_Probabilities[location_card,player_idx] - Unknown_Cards_Slack_Variables[i]==1)

    objective = -cp.sum(cp.log(Card_Probabilities))-cp.sum(cp.log(Card_Probabilities_Slack))
    if len(Unknown_Cards)>0:
        objective += -cp.sum(cp.log(Unknown_Cards_Slack_Variables))
    prob = cp.Problem(cp.Minimize(objective),constraints)
    prob.solve(solver='ECOS',max_iters=1000,verbose=False)
    return Card_Probabilities.value

class FormWidget(QWidget):
    def __init__(self, parent):        
        super(FormWidget, self).__init__(parent)
        self.create_table()
        self.layout = QGridLayout(self)

        Column1_y = 0

        self.layout.addWidget(self.table, Column1_y, 0, 1, N_Players+2)
        Column1_y += 1

        global Card_Counts
        if N_Players in [4,5]:
            self.card_choice = []
            if N_Players==4:
                Card_Possibilities = ['4','5']
            else: #N_Players==5
                Card_Possibilities = ['3','4']
            self.layout.addWidget(QLabel("Cards"),Column1_y,0)
            for i in range(N_Players):
                self.card_choice.append(QComboBox())
                self.card_choice[-1].addItems(Card_Possibilities)
                self.layout.addWidget(self.card_choice[-1],Column1_y,1+i)
                self.card_choice[-1].currentIndexChanged.connect(self.on_card_choice_change)
            self.Refresh_Card_Counts()
            for i in range(N_Players):
                if np.sum(Card_Counts)<18:
                    self.card_choice[i].setCurrentIndex(1)
                    self.Refresh_Card_Counts()
            Column1_y += 1
        else:
            Card_Counts = [3 if N_Players==6 else 6]*N_Players

        self.suspect_choice = QComboBox()
        self.suspect_choice.addItems(Card_Names[:6])
        self.layout.addWidget(self.suspect_choice,Column1_y,0)
        self.weapon_choice = QComboBox()
        self.weapon_choice.addItems(Card_Names[6:12])
        self.layout.addWidget(self.weapon_choice,Column1_y,1)
        self.room_choice = QComboBox()
        self.room_choice.addItems(Card_Names[12:N_Cards])
        self.layout.addWidget(self.room_choice,Column1_y,2)
        Column1_y += 1

        self.button_group = []
        for j in range(2):
            self.button_group.append(QButtonGroup(self.layout))
            self.layout.addWidget(QLabel("Refuted by" if j==1 else "Hypothesis by"),Column1_y,0)
            for i in range(1,N_Players+1+j):
                radio_button = QRadioButton("Player"+str(i) if i<N_Players+1 else "None")
                #radio_button.player_idx = i if i<N_Players+1 else -1
                self.button_group[-1].addButton(radio_button)
                self.layout.addWidget(self.button_group[-1].buttons()[-1], Column1_y, i)
            Column1_y += 1
            self.button_group[-1].buttons()[j].setChecked(True)

        self.button = QPushButton("Add Hypothesis")
        self.button.clicked.connect(self.add_hypothesis)
        self.layout.addWidget(self.button, Column1_y, 0)
        Column1_y += 1

        self.information_list = QListWidget()
        self.information_list.installEventFilter(self)
        self.layout.addWidget(self.information_list, 0, N_Players+2,1,2,Qt.AlignRight)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        self.layout.addWidget(self.undo_button, 1, N_Players+2)
        if len(Event_List)==0:
            self.undo_button.setEnabled(False)
        self.undo_shortcut = QShortcut(QtGui.QKeySequence('Ctrl+Z'), self)
        self.undo_shortcut.activated.connect(self.undo)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        self.layout.addWidget(self.redo_button, 1, N_Players+3)
        if len(Redo_List)==0:
            self.redo_button.setEnabled(False)
        self.redo_shortcut = QShortcut(QtGui.QKeySequence('Ctrl+Y'), self)
        self.redo_shortcut.activated.connect(self.redo)

        #Init initial state if it is not empty
        for event in Event_List:
            if len(event)==5:
                self.Display_Hypothesis(*event)
                self.process_hypothesis2(*event)
            else:
                self.Display_Known_Card(*event)
                Known_Cards.append(event)
        self.refresh_table()
        self.setLayout(self.layout)
    def create_table(self):
        self.table = QTableWidget(N_Cards,N_Players+1)
        self.table.setHorizontalHeaderLabels(['Player'+str(i) for i in range(1,N_Players+1)]+['Solution'])
        self.table.setVerticalHeaderLabels(Card_Names)
        for i in range(6):
            item = QTableWidgetItem(Card_Names[i])
            item.setBackground(QtGui.QColor(*Suspect_Colors[i]))
            self.table.setVerticalHeaderItem(i,item)
        self.table.installEventFilter(self)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        #self.table.selectionModel().currentColumnChanged.connect(self.on_column_selection)
        self.table.itemSelectionChanged.connect(self.on_table_selection)
        self.table.move(0,0)
        #self.suspect_weapon_line = QFrame(self.table.viewport())
        #self.suspect_weapon_line.setFrameShape(QFrame.HLine)
        #self.suspect_weapon_line.setFrameShadow(QFrame.Plain)
        #self.suspect_weapon_line.setFrameRect(QtCore.QRect(0,0,1000,100))
        #self.suspect_weapon_line.show()
    def Display_Known_Card(self,player_idx, card_idx, probability):
        posession_string = " has " if probability==1 else " doesn't have "
        self.information_list.addItem("P"+str(player_idx+1)+posession_string+Card_Names[card_idx])
    def Add_Known_Card_To_List(self,player_idx, card_idx, probability):
        self.Display_Known_Card(player_idx, card_idx, probability)
        Event_List.append([player_idx, card_idx, probability])
    def Display_Hypothesis(self,refutation_player_idx, suspect_card, weapon_card, location_card, hypothesis_player_idx):
        refuting_player_string = "P"+str(refutation_player_idx+1) if refutation_player_idx!=hypothesis_player_idx else "Noone"
        self.information_list.addItem(refuting_player_string+" refuted "+Card_Names[suspect_card]+"/"+Card_Names[weapon_card]+"/"+Card_Names[location_card]+" of P"+str(hypothesis_player_idx+1))
    def Add_Hypothesis_To_List(self,refutation_player_idx, suspect_card, weapon_card, location_card, hypothesis_player_idx):
        self.Display_Hypothesis(refutation_player_idx, suspect_card, weapon_card, location_card, hypothesis_player_idx)
        Event_List.append([refutation_player_idx, suspect_card, weapon_card, location_card, hypothesis_player_idx])
    def Remove_Information(self,idx):
        information = Event_List[idx]
        if len(information)==5:
            refutation_player_idx, suspect_card, weapon_card, location_card, hypothesis_player_idx = information
            passing_player_idx = (hypothesis_player_idx + 1) % N_Players
            while passing_player_idx!=refutation_player_idx:
                for card_idx in [suspect_card, weapon_card, location_card]:
                    Known_Cards.remove([passing_player_idx,card_idx,0])
                passing_player_idx = (passing_player_idx + 1) % N_Players
            if refutation_player_idx!=hypothesis_player_idx:
                Unknown_Cards.remove(information[:-1])
        else:
            Known_Cards.remove(information)
        self.information_list.takeItem(idx)
        Event_List.pop(idx)
        Redo_List.append(information)
        self.refresh_table()
        self.redo_button.setEnabled(True)
        if len(Event_List)==0:
            self.undo_button.setEnabled(False)
    def on_table_selection(self):
        model = self.table.selectionModel()
        cols = np.array([col.column() for col in model.selectedColumns() if col.column()<N_Players])
        rows = np.array([row.row() for row in model.selectedRows()])
        if len(cols)>0:
            self.button_group[0].buttons()[cols[0]].setChecked(True)
        if len(rows)>0:
            if np.any(rows<6):
                self.suspect_choice.setCurrentIndex(rows[rows<6][0])
            if np.any((rows<12) & (rows>=6)):
                self.weapon_choice.setCurrentIndex(rows[(rows<12) & (rows>=6)][0]-6)
            if np.any((rows<N_Cards) & (rows>=12)):
                self.room_choice.setCurrentIndex(rows[(rows<N_Cards) & (rows>=12)][0]-12)
    def refresh_table(self):
        probas = Find_Probabilities()
        if probas is not None:
            probabilities = np.round(probas,3)
            max_p = np.max(probabilities)
        for i in range(N_Cards):
            for j in range(N_Players+1):
                if probas is not None:
                    p = probabilities[i,j]
                    item = QTableWidgetItem(str(p))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(i,j,item)
                    self.table.item(i, j).setBackground(QtGui.QColor(int((1-p/max_p)*255),int((p/max_p)*255),0))
                else:
                    self.table.setItem(i,j,QTableWidgetItem(str(np.nan)))
                    self.table.item(i, j).setBackground(QtGui.QColor(255,0,0))
    def process_card(self,player_idx,card_idx):
        global Known_Cards
        information = [player_idx,card_idx,1]
        Known_Cards.append(information)
        self.Add_Known_Card_To_List(*information)
        self.refresh_table()
        self.undo_button.setEnabled(True)
    def add_card(self,player_idx,card_idx):
        self.process_card(player_idx,card_idx)
        self.table.clearSelection()
    def process_hypothesis2(self,refutation_player_idx, suspect_card, weapon_card, location_card, hypothesis_player_idx):
        passing_player_idx = (hypothesis_player_idx + 1) % N_Players
        while passing_player_idx!=refutation_player_idx:
            for card_idx in [suspect_card, weapon_card, location_card]:
                Known_Cards.append([passing_player_idx,card_idx,0])
            passing_player_idx = (passing_player_idx + 1) % N_Players
        information = [refutation_player_idx]+[suspect_card,weapon_card,location_card]+[hypothesis_player_idx]
        if refutation_player_idx!=hypothesis_player_idx:
            Unknown_Cards.append(information[:-1])
    def process_hypothesis(self,refutation_player_idx, suspect_card, weapon_card, location_card, hypothesis_player_idx):
        self.process_hypothesis2(refutation_player_idx, suspect_card, weapon_card, location_card, hypothesis_player_idx)
        self.Add_Hypothesis_To_List(refutation_player_idx, suspect_card, weapon_card, location_card, hypothesis_player_idx)
        self.undo_button.setEnabled(True)
        self.refresh_table()
    def add_hypothesis(self):
        global Known_Cards
        hypothesis_player_idx = [rdb.isChecked() for rdb in self.button_group[0].buttons()].index(True)
        selected_cards_idx = [self.suspect_choice.currentIndex(),self.weapon_choice.currentIndex()+6,self.room_choice.currentIndex()+12]
        refutation_player_idx = [rdb.isChecked() for rdb in self.button_group[1].buttons()].index(True)
        if refutation_player_idx == N_Players:
            refutation_player_idx = hypothesis_player_idx
        self.process_hypothesis(refutation_player_idx, *selected_cards_idx, hypothesis_player_idx)
        self.table.clearSelection()
    def valid_table_right_click(self,event):
        pos = self.table.viewport().mapFromGlobal(event.globalPos())
        return (self.table.itemAt(pos) is not None) and self.table.columnAt(pos.x())<N_Players
    def valid_information_right_click(self,event):
        pos = self.information_list.viewport().mapFromGlobal(event.globalPos())
        return self.information_list.itemAt(pos) is not None
    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.ContextMenu:
            menu = QtWidgets.QMenu()
            if (source is self.information_list) and self.valid_information_right_click(event):
                pos = self.information_list.viewport().mapFromGlobal(event.globalPos())
                menu.addAction('Remove')
                if menu.exec_(event.globalPos()):
                    idx = self.information_list.row(self.information_list.itemAt(pos))
                    self.Remove_Information(idx)
            elif (source is self.table) and self.valid_table_right_click(event):
                pos = self.table.viewport().mapFromGlobal(event.globalPos())
                menu.addAction('Add Card')
                if menu.exec_(event.globalPos()):
                    self.add_card(self.table.columnAt(pos.x()),self.table.rowAt(pos.y()))
            return True
        return super(FormWidget, self).eventFilter(source, event)
    def Refresh_Card_Counts(self):
        global Card_Counts
        Card_Counts = []
        for i in range(N_Players):
            Card_Counts.append(int(self.card_choice[i].currentText()))
    def on_card_choice_change(self):
        self.Refresh_Card_Counts()
        self.refresh_table()
    def undo(self):
        if len(Event_List)>0:
            self.Remove_Information(len(Event_List)-1)
    def redo(self):
        if len(Redo_List)>0:
            information = Redo_List[-1]
            if len(information)==5:
                self.process_hypothesis(*information)
            else:
                self.process_card(information[0],information[1])
            Redo_List.pop(-1)
            if len(Redo_List)==0:
                self.redo_button.setEnabled(False)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        screen_size = Get_Screen_Size()
        W = (525+100*(N_Players-3))+260
        H = 822 if N_Players in [4,5] else 791
        self.setGeometry((screen_size.width()-W)//2, (screen_size.height()-H)//2, W, H)
        self.setWindowTitle("Cluedo Sheet")
        self.form_widget = FormWidget(self) 
        self.setCentralWidget(self.form_widget) 
        self.show()

class GameDefinitionWidget(QWidget):
    def __init__(self, parent):        
        super(GameDefinitionWidget, self).__init__(parent)
        self.layout = QGridLayout(self)

        self.layout.addWidget(QLabel("Language"),0,0)
        self.language_dropdown = QComboBox()
        self.language_dropdown.addItems(Localization.keys())
        self.layout.addWidget(self.language_dropdown,0,1)

        self.layout.addWidget(QLabel("Player number"),1,0)
        self.player_number_dropdown = QComboBox()
        self.player_number_dropdown.addItems(['3','4','5','6'])
        self.layout.addWidget(self.player_number_dropdown,1,1)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play)
        self.layout.addWidget(self.play_button,2,0,1,2)
    def play(self):
        global N_Players, Card_Names, Suspect_Colors
        N_Players=int(self.player_number_dropdown.currentText())
        Chosen_Language = self.language_dropdown.currentText()
        Card_Names = Localization[Chosen_Language]['Card_Names']
        Suspect_Colors = Localization[Chosen_Language]['Suspect_Colors']
        self.parent().close()
        self.Open = Window()
        

class Player_Number_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        screen_size = Get_Screen_Size()
        W = 200
        H = 75
        self.setGeometry((screen_size.width()-W)//2, (screen_size.height()-H)//2, W, H)
        self.setWindowTitle("Game Definition")
        self.game_definition_widget = GameDefinitionWidget(self) 
        self.setCentralWidget(self.game_definition_widget) 
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Player_Number_Window()
    sys.exit(app.exec_())
