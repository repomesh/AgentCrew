from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from AgentCrew.modules.gui.themes import StyleProvider


class AgentListPanel(QWidget):
    """Composite widget for the agent list and its action buttons."""

    add_local_agent_requested = Signal()
    add_remote_agent_requested = Signal()
    import_requested = Signal()
    export_requested = Signal()
    remove_requested = Signal()
    selection_changed = Signal()
    agent_selected = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        style_provider = StyleProvider()

        layout = QVBoxLayout(self)

        self.agents_list = QListWidget()
        self.agents_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.agents_list.currentItemChanged.connect(self._on_current_item_changed)
        self.agents_list.itemSelectionChanged.connect(self._on_selection_changed)

        # Buttons
        buttons_layout = QHBoxLayout()

        self.add_agent_menu_btn = QPushButton("Add Agent")
        self.add_agent_menu_btn.setStyleSheet(
            style_provider.get_button_style("agent_menu")
        )
        add_agent_menu = QMenu(self)
        add_agent_menu.setStyleSheet(style_provider.get_agent_menu_style())
        add_local_action = add_agent_menu.addAction("Add Local Agent")
        add_remote_action = add_agent_menu.addAction("Add Remote Agent")
        self.add_agent_menu_btn.setMenu(add_agent_menu)

        add_local_action.triggered.connect(self.add_local_agent_requested)
        add_remote_action.triggered.connect(self.add_remote_agent_requested)

        self.import_agents_btn = QPushButton("Import")
        self.import_agents_btn.setStyleSheet(style_provider.get_button_style("green"))
        self.import_agents_btn.clicked.connect(self.import_requested)

        self.export_agents_btn = QPushButton("Export")
        self.export_agents_btn.setStyleSheet(style_provider.get_button_style("primary"))
        self.export_agents_btn.clicked.connect(self.export_requested)
        self.export_agents_btn.setEnabled(False)

        self.remove_agent_btn = QPushButton("Remove")
        self.remove_agent_btn.setStyleSheet(style_provider.get_button_style("red"))
        self.remove_agent_btn.clicked.connect(self._on_remove_clicked)
        self.remove_agent_btn.setEnabled(False)

        buttons_layout.addWidget(self.add_agent_menu_btn)
        buttons_layout.addWidget(self.import_agents_btn)
        buttons_layout.addWidget(self.export_agents_btn)
        buttons_layout.addWidget(self.remove_agent_btn)

        layout.addWidget(QLabel("Agents:"))
        layout.addWidget(self.agents_list)
        layout.addLayout(buttons_layout)

    def _on_current_item_changed(self, current, _):
        """Emit agent_selected with the current item (or None)."""
        self.agent_selected.emit(current)

    def _on_selection_changed(self):
        """Update button states and emit selection_changed."""
        self.update_selection_state()
        self.selection_changed.emit()

    def update_selection_state(self):
        """Update export/remove button enable states based on current selection."""
        has_selection = len(self.agents_list.selectedItems()) > 0
        self.export_agents_btn.setEnabled(has_selection)
        self.remove_agent_btn.setEnabled(has_selection)

    def _on_remove_clicked(self):
        """Confirm and remove selected agents."""
        self.remove_requested.emit()

    def load_agents(self, agents_config: dict):
        """Populate the list from a config dict."""
        self.agents_list.clear()

        local_agents = agents_config.get("agents", [])
        for agent_conf in local_agents:
            item_data = agent_conf.copy()
            item_data["agent_type"] = "local"
            item = QListWidgetItem(item_data.get("name", "Unnamed Local Agent"))
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            self.agents_list.addItem(item)

        remote_agents = agents_config.get("remote_agents", [])
        for agent_conf in remote_agents:
            item_data = agent_conf.copy()
            item_data["agent_type"] = "remote"
            item = QListWidgetItem(item_data.get("name", "Unnamed Remote Agent"))
            item.setData(Qt.ItemDataRole.UserRole, item_data)
            self.agents_list.addItem(item)

        self.agents_list.setCurrentRow(0)

    def get_selected_agents(self) -> list[dict]:
        """Return data dicts of selected items."""
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.agents_list.selectedItems()
        ]

    def add_agent(self, agent_data: dict):
        """Add a new agent item and select it."""
        item = QListWidgetItem(agent_data.get("name", "Unnamed Agent"))
        item.setData(Qt.ItemDataRole.UserRole, agent_data)
        self.agents_list.addItem(item)
        self.agents_list.setCurrentItem(item)

    def remove_selected_agents(self) -> bool:
        """Confirm removal of selected agents, remove them, and return success."""
        selected_items = self.agents_list.selectedItems()
        if not selected_items:
            return False

        if len(selected_items) == 1:
            agent_data = selected_items[0].data(Qt.ItemDataRole.UserRole)
            agent_name = agent_data.get("name", "this agent")
            message = f"Are you sure you want to delete the agent '{agent_name}'?"
        else:
            agent_names = [
                item.data(Qt.ItemDataRole.UserRole).get("name", "unnamed")
                for item in selected_items
            ]
            message = (
                f"Are you sure you want to delete {len(selected_items)} agents?\n\n• "
                + "\n• ".join(agent_names)
            )

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            rows_to_remove = sorted(
                [self.agents_list.row(item) for item in selected_items], reverse=True
            )
            for row in rows_to_remove:
                self.agents_list.takeItem(row)
            return True
        return False

    def set_enabled(self, enabled: bool):
        """Enable or disable action buttons."""
        self.add_agent_menu_btn.setEnabled(enabled)
        self.import_agents_btn.setEnabled(enabled)
        # export and remove are managed by selection state

    def find_agent_index_by_name(self, agent_name: str) -> int:
        """Find the index of an agent by name."""
        for i in range(self.agents_list.count()):
            item = self.agents_list.item(i)
            agent_data = item.data(Qt.ItemDataRole.UserRole)
            if agent_data.get("name", "") == agent_name:
                return i
        return -1
