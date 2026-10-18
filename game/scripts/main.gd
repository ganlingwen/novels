extends Node2D

const Rules = preload("res://scripts/rules.gd")
const World = preload("res://scripts/world.gd")
const SaveStore = preload("res://scripts/save_store.gd")
const MAP = Vector2(32, 150)
const C_BG = Color("#0b1219")
const C_PANEL = Color("#121f29")
const C_LINE = Color("#2a414a")
const C_TEXT = Color("#e4ece8")
const C_DIM = Color("#91aaa9")
const C_GREEN = Color("#7fe5b0")
const C_BLUE = Color("#8bc9ed")
const C_AMBER = Color("#eabc79")
const C_RED = Color("#ed7c75")
const QUALITY_COLORS = [C_DIM, C_GREEN, C_BLUE, Color("#d4a0ef")]

var character = Rules.new()
var store = SaveStore.new()
var world: RefCounted
var ui: Control
var font: SystemFont
var tabs: TabContainer
var status: Label
var health_bar: ProgressBar
var load_bar: ProgressBar
var stamina_bar: ProgressBar
var health_text: Label
var load_text: Label
var stamina_text: Label
var quest_text: Label
var log_text: Label
var skill_buttons: Array[Button] = []
var mode_button: Button
var base_button: Button
var pause_button: Button
var sheets: Array[VBoxContainer] = []
var save_enabled = true
var refresh_timer = 0.0
var save_timer = 0.0
var last_save = ""

func _ready() -> void:
	font = SystemFont.new()
	font.font_names = PackedStringArray(["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "sans-serif"])
	var message = store.load_character(character)
	save_enabled = not message.contains("禁止覆盖")
	world = World.new(character)
	world.note(message)
	create_ui()
	refresh_sheets()
	get_tree().auto_accept_quit = false

func panel_style(bg: Color, border: Color, radius: int = 8) -> StyleBoxFlat:
	var box = StyleBoxFlat.new()
	box.bg_color = bg
	box.border_color = border
	box.set_border_width_all(1)
	box.set_corner_radius_all(radius)
	box.content_margin_left = 12
	box.content_margin_right = 12
	box.content_margin_top = 8
	box.content_margin_bottom = 8
	return box

func create_ui() -> void:
	ui = Control.new()
	ui.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	ui.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var theme = Theme.new()
	theme.default_font = font
	theme.default_font_size = 16
	theme.set_color("font_color", "Label", C_TEXT)
	theme.set_color("font_color", "Button", C_TEXT)
	theme.set_color("font_disabled_color", "Button", Color("#57706f"))
	theme.set_stylebox("normal", "Button", panel_style(C_PANEL, C_LINE))
	theme.set_stylebox("hover", "Button", panel_style(Color("#203832"), C_GREEN))
	theme.set_stylebox("pressed", "Button", panel_style(Color("#284c3d"), C_GREEN))
	theme.set_stylebox("disabled", "Button", panel_style(Color("#111a20"), Color("#203039")))
	theme.set_stylebox("focus", "Button", StyleBoxEmpty.new())
	theme.set_stylebox("panel", "TabContainer", panel_style(C_PANEL, C_LINE))
	theme.set_stylebox("tab_selected", "TabContainer", panel_style(Color("#244035"), C_GREEN))
	theme.set_stylebox("tab_unselected", "TabContainer", panel_style(C_PANEL, C_LINE))
	theme.set_color("font_selected_color", "TabContainer", C_GREEN)
	theme.set_color("font_unselected_color", "TabContainer", C_DIM)
	ui.theme = theme
	add_child(ui)
	label_at("灰雾行者", Vector2(32, 22), Vector2(230, 45), 32, C_TEXT)
	label_at("GRAYMIST  /  单机战斗原型", Vector2(242, 38), Vector2(430, 25), 14, C_DIM)
	status = label_at("", Vector2(640, 36), Vector2(765, 30), 16, C_GREEN)
	status.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	health_bar = bar_at(Vector2(32, 100), Vector2(310, 12), C_GREEN)
	load_bar = bar_at(Vector2(364, 100), Vector2(310, 12), C_AMBER)
	stamina_bar = bar_at(Vector2(696, 100), Vector2(296, 12), C_BLUE)
	health_text = label_at("", Vector2(32, 73), Vector2(310, 24), 15, C_TEXT)
	load_text = label_at("", Vector2(364, 73), Vector2(310, 24), 15, C_TEXT)
	stamina_text = label_at("", Vector2(696, 73), Vector2(296, 24), 15, C_TEXT)
	label_at("猎鹰外围  /  废弃街区", Vector2(32, 121), Vector2(500, 25), 14, C_DIM)
	mode_button = button_at("", Vector2(1024, 82), Vector2(386, 47), cycle_mode)
	tabs = TabContainer.new()
	tabs.position = Vector2(1024, 150)
	tabs.size = Vector2(386, 590)
	tabs.get_tab_bar().focus_mode = Control.FOCUS_NONE
	ui.add_child(tabs)
	for title in ["行动", "属性", "天赋", "装备"]:
		var scroll = ScrollContainer.new()
		scroll.name = title
		scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
		tabs.add_child(scroll)
		var sheet = VBoxContainer.new()
		sheet.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		sheet.add_theme_constant_override("separation", 10)
		scroll.add_child(sheet)
		sheets.append(sheet)
	for i in 4:
		var index = i
		var button = button_at("", Vector2(32 + i * 190, 776), Vector2(178, 66),
			func(): world.cast(index))
		button.tooltip_text = Rules.SKILLS[i].desc + "\n积累 %d 短期负荷" % Rules.SKILLS[i].load
		skill_buttons.append(button)
	var potion = button_at("Q  急救药品", Vector2(792, 776), Vector2(200, 66), func(): world.drink())
	potion.tooltip_text = "冷却 8 秒；每次出发补满 5 份，仅本原型采用免费补给。"
	base_button = button_at("", Vector2(1024, 760), Vector2(386, 48), base_action)
	pause_button = button_at("Esc  暂停", Vector2(1024, 820), Vector2(185, 40), toggle_pause)
	button_at("保存进度", Vector2(1221, 820), Vector2(189, 40), save_game)
	label_at("WASD 移动  ·  左键攻击  ·  1–4 技能  ·  空格闪避  ·  Tab 模式  ·  E 基地", Vector2(32, 858),
		Vector2(980, 30), 15, C_DIM)
	label_at("程序美术 · 一级雷系 · 可随时暂停", Vector2(1024, 867), Vector2(390, 22), 13, C_DIM)

func label_at(text: String, pos: Vector2, extent: Vector2, size_value: int, color: Color) -> Label:
	var label = Label.new()
	label.text = text
	label.position = pos
	label.size = extent
	label.add_theme_font_size_override("font_size", size_value)
	label.add_theme_color_override("font_color", color)
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui.add_child(label)
	return label

func button_at(text: String, pos: Vector2, extent: Vector2, action: Callable) -> Button:
	var button = Button.new()
	button.text = text
	button.position = pos
	button.size = extent
	button.focus_mode = Control.FOCUS_NONE
	button.pressed.connect(action)
	ui.add_child(button)
	return button

func bar_at(pos: Vector2, extent: Vector2, color: Color) -> ProgressBar:
	var bar = ProgressBar.new()
	bar.position = pos
	bar.size = extent
	bar.show_percentage = false
	bar.add_theme_font_size_override("font_size", 1)
	var background = panel_style(Color("#1d2c33"), Color("#1d2c33"), 4)
	var fill = panel_style(color, color, 4)
	for style in [background, fill]:
		style.content_margin_top = 0
		style.content_margin_bottom = 0
	bar.add_theme_stylebox_override("background", background)
	bar.add_theme_stylebox_override("fill", fill)
	ui.add_child(bar)
	bar.size = extent
	return bar

func sheet_label(parent: VBoxContainer, text: String, color: Color = C_TEXT, size_value: int = 16) -> Label:
	var label = Label.new()
	label.text = text
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.add_theme_color_override("font_color", color)
	label.add_theme_font_size_override("font_size", size_value)
	parent.add_child(label)
	return label

func sheet_button(parent: VBoxContainer, text: String, action: Callable, disabled: bool = false) -> Button:
	var button = Button.new()
	button.text = text
	button.custom_minimum_size.y = 38
	button.focus_mode = Control.FOCUS_NONE
	button.disabled = disabled
	button.pressed.connect(action)
	parent.add_child(button)
	return button

func refresh_sheets() -> void:
	for sheet in sheets:
		for child in sheet.get_children():
			sheet.remove_child(child)
			child.queue_free()
	var sheet = sheets[0]
	sheet_label(sheet, "外围清理委托", C_GREEN, 22)
	quest_text = sheet_label(sheet, "")
	sheet_label(sheet, "从西侧入口出发，击倒丧尸后靠近取核。累计取核 10 枚将引来节点守卫。", C_DIM)
	sheet_label(sheet, "完成条件：12 枚晶核 + 守卫取核。\n返回入口交付，获得经验与稀有装备。", C_TEXT)
	sheet_label(sheet, "三种操作模式", C_GREEN, 19)
	sheet_label(sheet, "辅助：你移动，系统攻击与施法。\n自动：系统移动、取核、避险并返程。\n手动：鼠标选敌，左键与 1–4 施放。\n任何模式都可用 WASD 接管移动。", C_DIM)
	sheet_label(sheet, "行动记录", C_GREEN, 19)
	log_text = sheet_label(sheet, "", C_DIM, 15)
	sheet_label(sheet, "角色进度会自动保存。中途退出后从基地重新出发；本次战场不保留。", C_DIM, 13)
	sheet = sheets[1]
	var stats = character.stats()
	sheet_label(sheet, "Lv.%d  /  一级异能者" % character.level, C_GREEN, 22)
	sheet_label(sheet, "经验 %d / %s\n可用属性点 %d · 可用天赋点 %d" %
		[character.xp, str(character.xp_needed()) if character.level < 10 else "已满级",
		character.attribute_points, character.talent_points])
	var descriptions = {
		"body": "生命 +12，体力 +4",
		"strength": "武器伤害 +2.8",
		"agility": "提升普攻与闪避恢复",
		"spirit": "异能基础伤害 +3.2，护盾 +3"
	}
	for id in Rules.ATTRS:
		var key: String = id
		sheet_button(sheet, "%s  %d    ＋" % [Rules.ATTR_NAMES[id], character.attributes[id]],
			func(): allocate(key), world.active or character.attribute_points <= 0)
		sheet_label(sheet, descriptions[id], C_DIM, 14)
	sheet_label(sheet, "武器伤害 %.0f  ·  异能威力 %.0f\n护甲 %.0f  ·  攻击间隔 %.2f 秒\n移动速度 %.0f  ·  每秒消退负荷 %.1f" %
		[stats.power, stats.spell, stats.armor, stats.attack_cd, stats.speed, stats.cooling], C_TEXT)
	sheet_button(sheet, "推荐加点：异能生存", auto_allocate, world.active or character.attribute_points <= 0)
	sheet_button(sheet, "基地免费重置属性与天赋", respec, world.active)
	sheet_label(sheet, "在基地调整成长；可反复洗点试验流派。", C_DIM, 14)
	sheet = sheets[2]
	sheet_label(sheet, "天赋点  %d" % character.talent_points, C_GREEN, 22)
	var branch = ""
	for entry in Rules.TALENTS:
		if entry.branch != branch:
			branch = entry.branch
			sheet_label(sheet, branch, C_GREEN, 18)
		var id: String = entry.id
		var error: String = character.talent_error(id)
		var learned: bool = character.has_talent(id)
		sheet_button(sheet, ("✓ " if learned else "＋ ") + entry.name,
			func(): learn(id), world.active or not error.is_empty())
		var description: String = entry.desc
		if entry.has("requires"):
			description += "\n前置：" + character.talent(entry.requires).name
		sheet_label(sheet, description, C_DIM, 14)
	sheet = sheets[3]
	sheet_label(sheet, "装备与遗物", C_GREEN, 22)
	for slot in Rules.SLOTS:
		var equipped: Dictionary = character.equipped.get(slot, {})
		sheet_label(sheet, Rules.SLOT_NAMES[slot] + "  /  " +
			("空" if equipped.is_empty() else character.item_name(equipped)),
			C_DIM if equipped.is_empty() else QUALITY_COLORS[int(equipped.quality)], 15)
	sheet_label(sheet, "背包 %d / 120" % character.inventory.size(), C_GREEN, 19)
	sheet_label(sheet, "基地内换装。枪械暂用简化命中，无弹药管理。", C_DIM, 14)
	for i in character.inventory.size():
		var index = i
		var item: Dictionary = character.inventory[i]
		var slot: String = character.template(item.id).slot
		var equipped: bool = character.equipped.get(slot, {}) == item
		var button = sheet_button(sheet, ("✓ " if equipped else "") + character.item_name(item),
			func(): equip_item(index), world.active or equipped)
		button.add_theme_color_override("font_color", QUALITY_COLORS[int(item.quality)])
		sheet_label(sheet, describe_item(item), C_DIM, 14)
		if not equipped:
			sheet_button(sheet, "丢弃此物", func(): discard_item(index), world.active)
	world.changed = false

func describe_item(item: Dictionary) -> String:
	var names = {"power": "武器伤害", "hp": "生命", "armor": "护甲", "spell": "异能威力",
		"speed": "移动速度", "healing": "药品恢复", "cooling": "负荷消退/秒"}
	var parts: PackedStringArray = []
	var bonuses: Dictionary = character.item_stats(item)
	for key in bonuses:
		parts.append("%s +%.1f" % [names[key], bonuses[key]])
	var template: Dictionary = character.template(item.id)
	if template.has("reach"):
		parts.append("射程 %d" % template.reach)
	return " · ".join(parts)

func allocate(id: String) -> void:
	if not world.active and character.allocate(id):
		after_build_change()

func auto_allocate() -> void:
	if world.active:
		return
	while character.attribute_points > 0:
		character.allocate("body" if character.attributes.body * 2 < character.attributes.spirit else "spirit")
	after_build_change()

func learn(id: String) -> void:
	if not world.active and character.learn(id):
		after_build_change()

func equip_item(index: int) -> void:
	if not world.active and character.equip(index):
		after_build_change()

func discard_item(index: int) -> void:
	if world.active or index >= character.inventory.size():
		return
	var item: Dictionary = character.inventory[index]
	if character.equipped.get(character.template(item.id).slot, {}) == item:
		return
	character.inventory.remove_at(index)
	after_build_change()

func respec() -> void:
	if not world.active:
		character.respec()
		after_build_change()

func after_build_change() -> void:
	world.restore_health()
	refresh_sheets()
	save_game()

func cycle_mode() -> void:
	world.mode = (world.mode + 1) % 3
	world.note("切换为%s模式。" % World.MODES[world.mode])

func base_action() -> void:
	world.interact()
	refresh_sheets()

func toggle_pause() -> void:
	world.paused = not world.paused

func save_game() -> void:
	if not save_enabled:
		world.note("存档保护中；请先备份原存档。")
		return
	var error: Error = store.save_character(character)
	if error == OK:
		last_save = "已保存"
		world.save_requested = false
	else:
		last_save = "保存失败"
		world.note("保存失败：" + error_string(error))

func movement_input() -> Vector2:
	return Vector2(
		float(Input.is_physical_key_pressed(KEY_D) or Input.is_physical_key_pressed(KEY_RIGHT)) -
		float(Input.is_physical_key_pressed(KEY_A) or Input.is_physical_key_pressed(KEY_LEFT)),
		float(Input.is_physical_key_pressed(KEY_S) or Input.is_physical_key_pressed(KEY_DOWN)) -
		float(Input.is_physical_key_pressed(KEY_W) or Input.is_physical_key_pressed(KEY_UP)))

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		var key: int = event.physical_keycode
		if key == KEY_ESCAPE:
			toggle_pause()
		elif key == KEY_TAB:
			cycle_mode()
		elif key == KEY_E:
			base_action()
		elif key == KEY_Q:
			world.drink()
		elif key == KEY_SPACE:
			world.dash(movement_input())
		elif key >= KEY_1 and key <= KEY_4:
			var index = key - KEY_1
			world.cast(index, world.target_near_cursor(get_global_mouse_position() - MAP, Rules.SKILLS[index].reach))

func _process(delta: float) -> void:
	world.tick(minf(delta, 0.05), movement_input())
	var mouse = get_global_mouse_position() - MAP
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT) and Rect2(Vector2.ZERO, World.SIZE).has_point(mouse):
		world.attack(world.target_near_cursor(mouse, character.stats().reach))
	refresh_timer -= delta
	save_timer -= delta
	if world.changed:
		refresh_sheets()
	if refresh_timer <= 0:
		refresh_hud()
		refresh_timer = 0.1
	if world.save_requested and save_timer <= 0:
		save_game()
		save_timer = 2
	queue_redraw()

func refresh_hud() -> void:
	var stats = character.stats()
	status.text = "Lv.%d   /   雷系觉醒者     ·     %s     ·     %s" % [
		character.level, "行动中" if world.active else "基地整备", last_save]
	health_bar.max_value = stats.hp
	health_bar.value = world.hp
	health_text.text = "生命  %d / %d%s" % [ceili(world.hp), int(stats.hp),
		"  护盾 %d" % ceili(world.shield) if world.shield > 0 else ""]
	load_bar.value = world.load_value
	load_text.text = "短期负荷  %d / 100" % ceili(world.load_value)
	stamina_bar.max_value = stats.stamina
	stamina_bar.value = world.stamina
	stamina_text.text = "体力  %d / %d  ·  药品 %d" % [int(world.stamina), int(stats.stamina), world.potions]
	mode_button.text = "Tab  " + World.MODES[world.mode] + "模式  ↻"
	base_button.text = "E  出发探索" if not world.active else ("E  回城交付" if world.can_deliver() else "E  入口撤回基地")
	base_button.disabled = world.paused or (world.active and world.player.distance_to(World.BASE) >= 78)
	pause_button.text = "Esc  继续" if world.paused else "Esc  暂停"
	quest_text.text = ("取核 %d / 12    守卫：%s\n累计完成委托 %d 次" %
		[world.cores, "已消灭" if world.boss_taken else ("已出现" if world.boss_spawned else "未出现"), character.completions]
		if world.active else "基地已补满生命与 5 份药品。\n准备好后点击「出发探索」。\n累计完成委托 %d 次" % character.completions)
	log_text.text = "\n\n".join(world.messages.slice(0, 4))
	for i in 4:
		var skill: Dictionary = Rules.SKILLS[i]
		var cd: float = world.skill_cds[i]
		skill_buttons[i].text = "%s  %s\n%s" % [skill.key, skill.name,
			"%.1f 秒" % cd if cd > 0 else ("负荷过高" if world.load_value + skill.load > 100 else "就绪 · %d 负荷" % skill.load)]
		skill_buttons[i].disabled = not world.active or world.paused or cd > 0 or world.load_value + skill.load > 100

func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		save_game()
		get_tree().quit()

func text_at(text: String, pos: Vector2, size_value: int = 16, color: Color = C_TEXT) -> void:
	draw_string(font, pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1, size_value, color)

func _draw() -> void:
	if world == null:
		return
	draw_rect(Rect2(Vector2.ZERO, Vector2(1440, 900)), C_BG)
	draw_line(Vector2(32, 64), Vector2(1408, 64), C_LINE)
	draw_set_transform(MAP)
	draw_rect(Rect2(Vector2.ZERO, World.SIZE), Color("#142229"))
	# 街道、地块和杂物均为程序绘制，保持完整本地运行。
	draw_rect(Rect2(0, 260, 960, 110), Color("#1c3035"))
	draw_rect(Rect2(350, 0, 65, 600), Color("#1a2b31"))
	draw_rect(Rect2(790, 0, 70, 600), Color("#1a2b31"))
	for x in range(0, 960, 48):
		draw_line(Vector2(x, 0), Vector2(x, 600), Color(0.4, 0.6, 0.6, 0.035))
	for y in range(0, 600, 48):
		draw_line(Vector2(0, y), Vector2(960, y), Color(0.4, 0.6, 0.6, 0.035))
	for x in range(140, 940, 64):
		draw_line(Vector2(x, 315), Vector2(x + 25, 315), Color("#49615e"), 2)
	var titles = ["旧仓库", "配电房", "废弃商铺", "停车楼", "维修站"]
	for i in world.obstacles.size():
		var rect: Rect2 = world.obstacles[i]
		draw_rect(Rect2(rect.position + Vector2(7, 8), rect.size), Color("#0b171e"))
		draw_rect(rect, Color("#263b43"))
		draw_rect(rect.grow(-6), Color("#203139"))
		draw_rect(rect, Color("#486069"), false, 2)
		draw_line(rect.position + Vector2(10, 12), rect.position + Vector2(rect.size.x - 10, 12), C_LINE, 3)
		text_at(titles[i], rect.position + Vector2(14, rect.size.y / 2 + 5), 15, C_DIM)
	for i in 18:
		var p = Vector2(36 + (i * 137) % 895, 25 + (i * 83) % 545)
		if not world.blocked(p, 8):
			draw_line(p, p + Vector2(9, -4), Color("#3d5554"), 2)
			draw_line(p + Vector2(4, 6), p + Vector2(15, 4), Color("#2f4647"), 2)
	draw_rect(Rect2(12, 225, 110, 150), Color(0.35, 0.8, 0.6, 0.09))
	draw_rect(Rect2(12, 225, 110, 150), C_GREEN, false, 1)
	text_at("基地入口", Vector2(27, 247), 17, C_GREEN)
	text_at("E  交互", Vector2(34, 363), 14, C_DIM)
	for drop in world.drops:
		var p: Vector2 = drop.pos
		draw_rect(Rect2(p - Vector2(7, 7), Vector2(14, 14)), QUALITY_COLORS[int(drop.item.quality)])
		draw_rect(Rect2(p - Vector2(4, 4), Vector2(8, 8)), C_BG)
	for enemy in world.enemies:
		draw_enemy(enemy)
	draw_player()
	for effect in world.effects:
		draw_effect(effect)
	draw_rect(Rect2(Vector2.ZERO, World.SIZE), C_LINE, false, 2)
	if not world.active:
		draw_rect(Rect2(220, 172, 550, 220), Color(0.04, 0.08, 0.10, 0.96))
		draw_rect(Rect2(220, 172, 550, 220), C_LINE, false, 1)
		text_at("灰雾仍在街道尽头。", Vector2(253, 224), 28, C_TEXT)
		text_at("整备装备与天赋，前往街区完成清理委托。", Vector2(253, 268), 18, C_DIM)
		text_at("第一次玩？按 Tab 切到自动模式，再按 E 出发。", Vector2(253, 306), 17, C_GREEN)
		text_at("原型采用宽松规则：失败保留成长，基地免费补给。", Vector2(253, 350), 15, C_DIM)
	if world.paused:
		draw_rect(Rect2(Vector2.ZERO, World.SIZE), Color(0.02, 0.04, 0.06, 0.7))
		text_at("已暂停", Vector2(405, 282), 36, C_GREEN)
		text_at("按 Esc 继续", Vector2(416, 324), 18, C_TEXT)
	draw_set_transform(Vector2.ZERO)

func draw_player() -> void:
	var p: Vector2 = world.player
	draw_ellipse_shadow(p, 18)
	if world.shield > 0:
		draw_arc(p, 26, 0, TAU, 40, C_BLUE, 2, true)
	if world.dash_time > 0:
		draw_line(p - world.facing * 38, p, Color(0.5, 0.9, 0.75, 0.6), 10, true)
	draw_circle(p, 17, Color("#15352d"))
	draw_circle(p, 12, C_GREEN)
	draw_circle(p + Vector2(0, -5), 6, Color("#dbe9d5"))
	var side = world.facing.rotated(PI / 2)
	var points = PackedVector2Array([p + world.facing * 29, p + world.facing * 17 + side * 5,
		p + world.facing * 17 - side * 5])
	draw_colored_polygon(points, C_TEXT)
	text_at("行者", p + Vector2(-15, -26), 13, C_GREEN)

func draw_ellipse_shadow(p: Vector2, radius: float) -> void:
	draw_set_transform(MAP + p + Vector2(0, 10), 0, Vector2(1, 0.45))
	draw_circle(Vector2.ZERO, radius, Color(0, 0, 0, 0.35))
	draw_set_transform(MAP)

func draw_enemy(enemy: Dictionary) -> void:
	var p: Vector2 = enemy.pos
	var info: Dictionary = World.ENEMY_TYPES[enemy.type]
	var radius: float = info.radius
	var colors = [Color("#90a47c"), Color("#b6a572"), Color("#a58973"), Color("#ce9dca"), C_RED]
	var color: Color = colors[enemy.type]
	if enemy.windup > 0:
		draw_circle(enemy.blast, 115, Color(0.9, 0.3, 0.27, 0.13))
		draw_arc(enemy.blast, 115, 0, TAU, 64, C_RED, 2, true)
		draw_arc(enemy.blast, 115 * (1 - enemy.windup / 1.15), 0, TAU, 64, C_AMBER, 2, true)
		text_at("冲击预警", enemy.blast + Vector2(-35, -126), 14, C_RED)
	if enemy.down:
		draw_line(p - Vector2(14, 5), p + Vector2(14, 5), Color("#6b6660"), 7, true)
		draw_circle(p, 4 + sin(world.clock * 5), C_BLUE)
		text_at("靠近取核", p + Vector2(-26, -16), 12, C_DIM)
		return
	draw_ellipse_shadow(p, radius + 3)
	if enemy.flash > 0:
		color = C_TEXT
	if enemy.type == 2 or enemy.type == 4:
		draw_rect(Rect2(p - Vector2(radius, radius), Vector2(radius * 2, radius * 2)), color)
		draw_rect(Rect2(p - Vector2(radius - 5, radius - 5), Vector2((radius - 5) * 2, (radius - 5) * 2)), color.darkened(0.4))
	else:
		draw_circle(p, radius, color)
		draw_circle(p, radius - 5, color.darkened(0.4))
	var direction = p.direction_to(world.player)
	draw_circle(p + direction * radius * 0.55, 3, Color("#f4cdaf"))
	if enemy.type >= 3:
		draw_arc(p, radius + 5, world.clock, world.clock + PI * 1.5, 20, color, 1.5, true)
		text_at(info.name, p + Vector2(-32, -radius - 17), 14, color)
	var width = 55 if enemy.type < 4 else 90
	draw_rect(Rect2(p + Vector2(-width / 2.0, -radius - 9), Vector2(width, 4)), C_BG)
	draw_rect(Rect2(p + Vector2(-width / 2.0, -radius - 9), Vector2(width * enemy.hp / enemy.max_hp, 4)), color)
	if enemy.stun > 0:
		text_at("麻痹", p + Vector2(-14, radius + 17), 12, C_BLUE)

func draw_effect(effect: Dictionary) -> void:
	match effect.kind:
		"number":
			text_at(effect.text, effect.pos + Vector2(-8, -28 - (0.7 - effect.time) * 38), 20,
				C_BLUE if effect.electric else C_TEXT)
		"bolt":
			var direction: Vector2 = (effect.to - effect.from).orthogonal().normalized()
			var points = PackedVector2Array()
			for i in 7:
				points.append(effect.from.lerp(effect.to, i / 6.0) + direction * (0 if i in [0, 6] else sin(i * 8.3) * 12))
			draw_polyline(points, Color(0.4, 0.7, 1.0, 0.18), 8, true)
			draw_polyline(points, C_BLUE, 2, true)
		"shot":
			draw_line(effect.from, effect.to, C_AMBER, 2, true)
		"slash":
			var angle: float = (effect.to - effect.from).angle()
			draw_arc(effect.from, 48, angle - 0.7, angle + 0.7, 16, C_TEXT, 4, true)
		"ring", "blast":
			draw_arc(effect.pos, effect.radius * (1 - effect.time / 0.4), 0, TAU, 48,
				C_RED if effect.kind == "blast" else C_BLUE, 3, true)
		"hurt":
			draw_arc(effect.pos, 22, 0, TAU, 24, C_RED, 3, true)
