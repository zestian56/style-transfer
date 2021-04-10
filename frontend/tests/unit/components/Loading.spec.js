import { shallowMount } from "@vue/test-utils";
import Loading from "@/components/Loading.vue";

describe("Loading.vue", () => {
  it("renders props.msg when passed", () => {
    const msg = "new message";
    const wrapper = shallowMount(Loading, {
      props: { msg },
    });
    expect(wrapper).toMatchSnapshot();
  });
});
