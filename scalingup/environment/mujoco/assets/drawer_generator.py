template_xml = """
<mujoco model="drawer">
  <compiler angle="radian" autolimits="true" />
  <option impratio="10" />
  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" />
    </default>
    <default class="collision">
      <geom type="mesh" group="3" />
    </default>
  </default>

  <asset>
    <material name="drawer_material" rgba="{rgba}" />
  </asset>
  <worldbody>
    <body name="drawer">
      <geom size="{width_plus_three_thickness} {thickness} {half_total_height}" pos="0 -{depth_plus_two_thickness} {half_total_height}" class="visual" type="box" material="drawer_material" />
      <geom size="{width_plus_three_thickness} {thickness} {half_total_height}" pos="0 {depth_plus_two_thickness} {half_total_height}" class="visual" type="box" material="drawer_material" />
      <geom size="{width_plus_three_thickness} {depth_plus_three_thickness} {thickness}" pos="0 0 {thickness}" class="visual" type="box" material="drawer_material" />
      <geom size="{width_plus_three_thickness} {depth_plus_three_thickness} {thickness}" pos="0 0 {total_height}" class="visual" type="box" material="drawer_material" />
      <geom size="{thickness} {depth_plus_three_thickness} {half_total_height}" pos="-{width_plus_two_thickness} 0 {half_total_height}" class="visual" type="box" material="drawer_material" />


      <geom size="{width_plus_three_thickness} {thickness} {half_total_height}" pos="0 -{depth_plus_two_thickness} {half_total_height}" class="collision" type="box" />
      <geom size="{width_plus_three_thickness} {thickness} {half_total_height}" pos="0 {depth_plus_two_thickness} {half_total_height}" class="collision" type="box" />
      <geom size="{width_plus_three_thickness} {depth_plus_three_thickness} {thickness}" pos="0 0 {thickness}" class="collision" type="box" />
      <geom size="{width_plus_three_thickness} {depth_plus_three_thickness} {thickness}" pos="0 0 {total_height}" class="collision" type="box" />
      <geom size="{thickness} {depth_plus_three_thickness} {half_total_height}" pos="-{width_plus_two_thickness} 0 {half_total_height}" class="collision" type="box" />


      <body name="bottom_drawer" pos="{three_thickness} 0 {bottom_drawer_z_pos}">
        <joint name="bottom_drawer_slide" type="slide" axis="1 0 0" limited="true" range="0 {drawer_range}"
          damping="50.0" frictionloss="1.0" />

        <!-- bottom -->
        <geom size="{width} {depth} {thickness}" class="visual" type="box" pos="0 0 -{height_minus_thickness}" material="drawer_material" />
        <!-- left and right -->
        <geom size="{width} {thickness} {height}" class="visual" type="box" pos="0 {depth_minus_thickness} 0" material="drawer_material" />
        <geom size="{width} {thickness} {height}" class="visual" type="box" pos="0 -{depth_minus_thickness} 0" material="drawer_material" />
        <!-- front and back -->
        <geom size="{thickness} {depth} {height}" class="visual" type="box" pos="{width_minus_thickness} 0 0" material="drawer_material" />
        <geom size="{thickness} {depth} {height}" class="visual" type="box" pos="-{width_minus_thickness} 0 0" material="drawer_material" />

        <!-- bottom -->
        <geom size="{width} {depth} {thickness}" class="collision" type="box" pos="0 0 -{height_minus_thickness}" />
        <!-- left and right -->
        <geom size="{width} {thickness} {height}" class="collision" type="box" pos="0 {depth_minus_thickness} 0" />
        <geom size="{width} {thickness} {height}" class="collision" type="box" pos="0 -{depth_minus_thickness} 0" />
        <!-- front and back -->
        <geom size="{thickness} {depth} {height}" class="collision" type="box" pos="{width_minus_thickness} 0 0" />
        <geom size="{thickness} {depth} {height}" class="collision" type="box" pos="-{width_minus_thickness} 0 0" />

        <body name="bottom_drawer_handle" pos="{handle_x_pos} 0 0" euler="1.57 0 0">
          <geom size="{handle_radius} {handle_length}" class="visual" type="cylinder" material="drawer_material" />
          <geom size="{handle_radius} {handle_length}" class="collision" type="cylinder" />
        </body>
      </body>
      <body name="middle_drawer" pos="{three_thickness} 0 {middle_drawer_z_pos}">
        <joint name="middle_drawer_slide" type="slide" axis="1 0 0" limited="true" range="0 {drawer_range}"
          damping="50.0" frictionloss="1.0" />

        <!-- bottom -->
        <geom size="{width} {depth} {thickness}" class="visual" type="box" pos="0 0 -{height_minus_thickness}" material="drawer_material" />
        <!-- left and right -->
        <geom size="{width} {thickness} {height}" class="visual" type="box" pos="0 {depth_minus_thickness} 0" material="drawer_material" />
        <geom size="{width} {thickness} {height}" class="visual" type="box" pos="0 -{depth_minus_thickness} 0" material="drawer_material" />
        <!-- front and back -->
        <geom size="{thickness} {depth} {height}" class="visual" type="box" pos="{width_minus_thickness} 0 0" material="drawer_material" />
        <geom size="{thickness} {depth} {height}" class="visual" type="box" pos="-{width_minus_thickness} 0 0" material="drawer_material" />

        <!-- bottom -->
        <geom size="{width} {depth} 0.01" class="collision" type="box" pos="0 0 -{height_minus_thickness}" />
        <!-- left and right -->
        <geom size="{width} 0.01 {height}" class="collision" type="box" pos="0 {depth_minus_thickness} 0" />
        <geom size="{width} 0.01 {height}" class="collision" type="box" pos="0 -{depth_minus_thickness} 0" />
        <!-- front and back -->
        <geom size="0.01 {depth} {height}" class="collision" type="box" pos="{width_minus_thickness} 0 0" />
        <geom size="0.01 {depth} {height}" class="collision" type="box" pos="-{width_minus_thickness} 0 0" />

        <body name="middle_drawer_handle" pos="{handle_x_pos} 0 0" euler="1.57 0 0">
          <geom size="{handle_radius} {handle_length}" class="visual" type="cylinder" material="drawer_material" />
          <geom size="{handle_radius} {handle_length}" class="collision" type="cylinder" />
        </body>
      </body>
      <body name="top_drawer" pos="{three_thickness} 0 {top_drawer_z_pos}">
        <joint name="top_drawer_slide" type="slide" axis="1 0 0" limited="true" range="0 {drawer_range}"
          damping="50.0" frictionloss="1.0" />

        <!-- bottom -->
        <geom size="{width} {depth} {thickness}" class="visual" type="box" pos="0 0 -{height_minus_thickness}" material="drawer_material" />
        <!-- left and right -->
        <geom size="{width} {thickness} {height}" class="visual" type="box" pos="0 {depth_minus_thickness} 0" material="drawer_material" />
        <geom size="{width} {thickness} {height}" class="visual" type="box" pos="0 -{depth_minus_thickness} 0" material="drawer_material" />
        <!-- front and back -->
        <geom size="{thickness} {depth} {height}" class="visual" type="box" pos="{width_minus_thickness} 0 0" material="drawer_material" />
        <geom size="{thickness} {depth} {height}" class="visual" type="box" pos="-{width_minus_thickness} 0 0" material="drawer_material" />

        <!-- bottom -->
        <geom size="{width} {depth} {thickness}" class="collision" type="box" pos="0 0 -{height_minus_thickness}" />
        <!-- left and right -->
        <geom size="{width} {thickness} {height}" class="collision" type="box" pos="0 {depth_minus_thickness} 0" />
        <geom size="{width} {thickness} {height}" class="collision" type="box" pos="0 -{depth_minus_thickness} 0" />
        <!-- front and back -->
        <geom size="{thickness} {depth} {height}" class="collision" type="box" pos="{width_minus_thickness} 0 0" />
        <geom size="{thickness} {depth} {height}" class="collision" type="box" pos="-{width_minus_thickness} 0 0" />

        <body name="top_drawer_handle" pos="{handle_x_pos} 0 0" euler="1.57 0 0">
          <geom size="{handle_radius} {handle_length}" class="visual" type="cylinder" material="drawer_material" />
          <geom size="{handle_radius} {handle_length}" class="collision" type="cylinder" />
        </body>
      </body>
    </body>

  </worldbody>
</mujoco>
"""  # noqa: B950


def create_drawer(
    height_per_drawer: float = 0.08,
    width: float = 0.17,
    depth: float = 0.17,
    handle_radius: float = 0.01,
    handle_length: float = 0.05,
    handle_protrusion: float = 0.03,
    thickness: float = 0.01,
    num_drawers: int = 3,
) -> str:
    assert num_drawers == 3
    return template_xml.format(
        rgba="0.3 0.2 0.05 1",
        height=height_per_drawer,
        thickness=thickness,
        width=width,
        depth=depth,
        handle_radius=handle_radius,
        handle_length=handle_length,
        handle_x_pos=width + handle_protrusion,
        height_minus_thickness=height_per_drawer - thickness,
        depth_minus_thickness=depth - thickness,
        width_minus_thickness=width - thickness,
        half_total_height=num_drawers * height_per_drawer + thickness * 4,
        total_height=(num_drawers * height_per_drawer + thickness * 4) * 2,
        width_plus_two_thickness=width + thickness * 2,
        depth_plus_two_thickness=depth + thickness * 2,
        width_plus_three_thickness=width + thickness * 3,
        depth_plus_three_thickness=depth + thickness * 3,
        top_drawer_z_pos=height_per_drawer * 5 + thickness * 5,
        middle_drawer_z_pos=height_per_drawer * 3 + thickness * 4,
        bottom_drawer_z_pos=height_per_drawer * 1 + thickness * 3,
        drawer_range=width * 2,
        three_thickness=thickness * 3,
    )
